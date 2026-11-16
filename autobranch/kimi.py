"""Client for the Kimi reasoning model behind a Modal Shared Endpoint.

Stdlib only -- `urllib.request`, no `openai`, no `requests`. Agents run inside
whatever image the environment bundle asked for, and that image is not ours to
add dependencies to, so this module must import in a bare sandbox.

It is a reasoning model: the response carries `content` and
`reasoning_content`, and reasoning comes first. At `max_tokens=16` all 16
completion tokens were reasoning and `content` came back empty. An empty
`content` is almost never "nothing to say", it is "the budget ran out
mid-thought", so it raises rather than returning silence.
"""

import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, replace
from typing import Any
from uuid import uuid4

DEFAULT_BASE_URL = "https://sf-hackathon--ep-autobranch-kimi-server.us-west.modal.direct"
DEFAULT_MODEL = "moonshotai/Kimi-K3.1-Pro"
CONTEXT_WINDOW = 1_048_576

# Measured endpoint pricing, USD per million tokens. Reasoning tokens are a
# subset of `completion_tokens` and bill at the output rate; cached prompt
# tokens are a subset of `prompt_tokens` and bill at a tenth of fresh input.
USD_PER_MTOK_INPUT = 2.00
USD_PER_MTOK_INPUT_CACHED = 0.20
USD_PER_MTOK_OUTPUT = 10.00

RETRY_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})
_BACKOFF_BASE_SECONDS = 0.5
_BACKOFF_CAP_SECONDS = 30.0


def _sleep(seconds: float) -> None:
    """Indirection so tests can run the backoff path without waiting."""
    time.sleep(seconds)


class KimiError(RuntimeError):
    """Any failure talking to the endpoint, with the response body attached."""

class KimiTruncated(KimiError):
    """The model spent its token budget reasoning and never produced content.

    Retryable, and retried once automatically at double `max_tokens`. Distinct
    from a successful empty answer on purpose -- see the module docstring.
    """

class KimiRateLimited(KimiError):
    """HTTP 429. Retried with exponential backoff and jitter."""

@dataclass(frozen=True)
class Usage:
    """Token accounting for one call, or a running total when summed.

    `reasoning_tokens` ⊆ `completion_tokens` and `cached_tokens` ⊆
    `prompt_tokens`, matching the OpenAI `*_tokens_details` convention the
    endpoint follows. Keeping them as subsets rather than disjoint buckets is
    what makes `cost_usd` a straight translation of the price sheet.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0

    @property
    def cost_usd(self) -> float:
        fresh_prompt = max(0, self.prompt_tokens - self.cached_tokens)
        return (
            fresh_prompt * USD_PER_MTOK_INPUT
            + self.cached_tokens * USD_PER_MTOK_INPUT_CACHED
            + self.completion_tokens * USD_PER_MTOK_OUTPUT
        ) / 1_000_000

    @property
    def reasoning_fraction(self) -> float:
        """Share of output spent thinking -- the number that sets `max_tokens`."""
        if self.completion_tokens <= 0:
            return 0.0
        return self.reasoning_tokens / self.completion_tokens

    def __add__(self, other: "Usage") -> "Usage":
        if not isinstance(other, Usage):
            return NotImplemented
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
        )

    @staticmethod
    def parse(raw: Any) -> "Usage":
        raw = raw or {}
        prompt_details = raw.get("prompt_tokens_details") or {}
        completion_details = raw.get("completion_tokens_details") or {}
        return Usage(
            prompt_tokens=int(raw.get("prompt_tokens") or 0),
            completion_tokens=int(raw.get("completion_tokens") or 0),
            reasoning_tokens=int(completion_details.get("reasoning_tokens") or 0),
            cached_tokens=int(
                prompt_details.get("cached_tokens")
                if prompt_details.get("cached_tokens") is not None
                else raw.get("cached_tokens") or 0
            ),
        )


@dataclass(frozen=True)
class Reply:
    """One assistant turn, with thinking kept separate from the answer.

    `reasoning` is retained rather than discarded because it is the only
    explanation of *why* a branch mutated a reward the way it did, and
    `losers.json` is the scientific output of a run.
    """

    content: str
    reasoning: str
    finish_reason: str
    usage: Usage
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


def extract_json_object(text: str) -> str:
    """Return the outermost balanced `{...}` in `text`.

    A reasoning model asked for JSON still wraps it in ``` fences, prefaces it
    with "Here is the action:", or trails a sentence after it. Brace matching
    with string awareness survives all three and costs less than a regex that
    pretends to parse JSON.
    """
    start = text.find("{")
    if start < 0:
        raise ValueError("no '{' in response")
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError("unbalanced braces: object never closes")


def _echo(text: str, limit: int = 400) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [+{len(text) - limit} chars]"


class Kimi:
    """A conversation-sticky, budget-aware chat client.

    Not thread-safe: one instance per agent loop, which is also the unit that
    owns an affinity key and a usage ledger.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        token_id: str | None = None,
        token_secret: str | None = None,
        model: str = DEFAULT_MODEL,
        affinity_key: str | None = None,
        default_max_tokens: int = 4096,
        timeout: float = 300.0,
        max_retries: int = 5,
        opener: Any | None = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get("AUTOBRANCH_KIMI_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.token_id = token_id or os.environ.get("MODAL_PROXY_TOKEN_ID") or ""
        self.token_secret = token_secret or os.environ.get("MODAL_PROXY_TOKEN_SECRET") or ""
        if not self.token_id or not self.token_secret:
            raise KimiError(
                "missing proxy auth: set MODAL_PROXY_TOKEN_ID and MODAL_PROXY_TOKEN_SECRET "
                "(in a Modal container, attach the secret 'autobranch-kimi-auth')"
            )
        self.model = model
        # Always send *some* affinity key: a client with none would scatter its
        # turns across workers and pay full price for its own transcript.
        self.affinity_key = affinity_key or f"autobranch-{uuid4().hex[:16]}"
        self.default_max_tokens = int(default_max_tokens)
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self._opener = opener if opener is not None else urllib.request.build_opener()
        self.total_usage = Usage()
        self.calls = 0

    # ---- public API ----------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        *,
        max_tokens: int | None = None,
        temperature: float = 0.6,
        json_object: bool = False,
        affinity_key: str | None = None,
    ) -> Reply:
        """One completion, with transport retries and one truncation retry.

        `KimiTruncated` escapes only after the doubled-budget attempt also
        failed to produce content; at that point the prompt, not the budget, is
        the problem and the caller should know.
        """
        budget = int(max_tokens or self.default_max_tokens)
        key = affinity_key or self.affinity_key
        try:
            return self._chat_with_retries(
                messages, max_tokens=budget, temperature=temperature,
                json_object=json_object, affinity_key=key,
            )
        except KimiTruncated:
            doubled = min(budget * 2, CONTEXT_WINDOW)
            if doubled <= budget:
                raise
            return self._chat_with_retries(
                messages, max_tokens=doubled, temperature=temperature,
                json_object=json_object, affinity_key=key,
            )

    def chat_json(self, messages: list[dict], *, schema_hint: str | None = None, **kw: Any) -> dict:
        """Chat and parse a single JSON object out of the answer."""
        if schema_hint:
            messages = list(messages) + [
                {
                    "role": "user",
                    "content": (
                        "Reply with a single JSON object and nothing else. Expected shape:\n"
                        + schema_hint
                    ),
                }
            ]
        kw.setdefault("json_object", True)
        reply = self.chat(messages, **kw)
        return self.parse_json_reply(reply)

    @staticmethod
    def parse_json_reply(reply: Reply) -> dict:
        """Parse `reply.content` as a JSON object, erroring with the body echoed.

        Separate from `chat_json` so callers that need the `Reply` itself (for
        usage, or to put the reasoning in a transcript) can reuse the exact
        same tolerance for fences and prose.
        """
        try:
            blob = extract_json_object(reply.content)
        except ValueError as exc:
            raise KimiError(
                f"expected a JSON object, found none ({exc}); "
                f"finish_reason={reply.finish_reason!r} content={_echo(reply.content)!r}"
            ) from exc
        try:
            parsed = json.loads(blob)
        except json.JSONDecodeError as exc:
            raise KimiError(
                f"malformed JSON at char {exc.pos} ({exc.msg}); body={_echo(blob)!r}"
            ) from exc
        if not isinstance(parsed, dict):
            raise KimiError(f"expected a JSON object, got {type(parsed).__name__}; body={_echo(blob)!r}")
        return parsed

    # ---- internals -----------------------------------------------------

    def _chat_with_retries(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float,
        json_object: bool,
        affinity_key: str,
    ) -> Reply:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_object:
            payload["response_format"] = {"type": "json_object"}

        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                raw = self._post("/v1/chat/completions", payload, affinity_key=affinity_key)
            except KimiRateLimited as exc:
                last = exc
                if attempt >= self.max_retries:
                    break
                _sleep(self._backoff(attempt, getattr(exc, "retry_after", None)))
                continue
            except KimiError as exc:
                if not getattr(exc, "retryable", False) or attempt >= self.max_retries:
                    raise
                last = exc
                _sleep(self._backoff(attempt, None))
                continue
            return self._to_reply(raw, max_tokens=max_tokens)
        raise last if last is not None else KimiError("retries exhausted with no error recorded")

    def _backoff(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return min(retry_after, _BACKOFF_CAP_SECONDS)
        ceiling = min(_BACKOFF_BASE_SECONDS * (2**attempt), _BACKOFF_CAP_SECONDS)
        # Decorrelated jitter: a synchronous retry storm from eight sibling
        # branches is exactly how a 429 becomes a sustained 429.
        return random.uniform(ceiling * 0.5, ceiling)

    def _post(self, path: str, payload: dict, *, affinity_key: str) -> dict:
        body = json.dumps(payload).encode()
        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Modal-Key": self.token_id,
                "Modal-Secret": self.token_secret,
                "Modal-Routing-Affinity-Key": affinity_key,
            },
        )
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                text = response.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            detail = _echo(_read_error_body(exc))
            if exc.code == 429:
                err = KimiRateLimited(f"HTTP 429 rate limited: {detail}")
                err.retryable = True
                err.retry_after = _retry_after(exc)
                raise err from exc
            err = KimiError(f"HTTP {exc.code} {exc.reason}: {detail}")
            err.retryable = exc.code in RETRY_STATUS
            raise err from exc
        except urllib.error.URLError as exc:
            err = KimiError(f"transport error talking to {self.base_url}: {exc.reason}")
            err.retryable = True  # a dropped connection mid-run is not a dead run
            raise err from exc
        except TimeoutError as exc:
            err = KimiError(f"timeout after {self.timeout}s talking to {self.base_url}")
            err.retryable = True
            raise err from exc

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise KimiError(f"endpoint returned non-JSON: {_echo(text)!r}") from exc
        if not isinstance(parsed, dict):
            raise KimiError(f"endpoint returned {type(parsed).__name__}, expected an object")
        if "error" in parsed and "choices" not in parsed:
            raise KimiError(f"endpoint error: {_echo(json.dumps(parsed['error']))}")
        return parsed

    def _to_reply(self, raw: dict, *, max_tokens: int) -> Reply:
        choices = raw.get("choices") or []
        if not choices:
            raise KimiError(f"response had no choices: {_echo(json.dumps(raw))}")
        choice = choices[0] or {}
        message = choice.get("message") or {}
        content = (message.get("content") or "").strip()
        reasoning = (message.get("reasoning_content") or message.get("reasoning") or "").strip()
        finish_reason = str(choice.get("finish_reason") or "")
        usage = Usage.parse(raw.get("usage"))

        # Bill before raising: a truncated call still cost output tokens, and a
        # budget that ignores failed calls is a budget that overspends.
        self.total_usage = self.total_usage + usage
        self.calls += 1

        reply = Reply(
            content=content, reasoning=reasoning, finish_reason=finish_reason,
            usage=usage, raw=raw,
        )
        if not content and (finish_reason == "length" or reasoning):
            raise KimiTruncated(
                f"empty content after {usage.completion_tokens} completion tokens "
                f"({usage.reasoning_tokens} reasoning) with max_tokens={max_tokens}, "
                f"finish_reason={finish_reason!r}; the model never stopped thinking. "
                f"reasoning tail={_echo(reasoning[-200:], 200)!r}"
            )
        return reply

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            f"Kimi(model={self.model!r}, affinity_key={self.affinity_key!r}, "
            f"calls={self.calls}, spent=${self.total_usage.cost_usd:.4f})"
        )


def _read_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", "replace")
    except Exception:  # pragma: no cover - body already consumed or absent
        return ""


def _retry_after(exc: urllib.error.HTTPError) -> float | None:
    raw = None
    headers = getattr(exc, "headers", None)
    if headers is not None:
        raw = headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


def usage_minus(a: Usage, b: Usage) -> Usage:
    """`a - b`, for turning two ledger snapshots into the spend between them."""
    return replace(
        a,
        prompt_tokens=a.prompt_tokens - b.prompt_tokens,
        completion_tokens=a.completion_tokens - b.completion_tokens,
        reasoning_tokens=a.reasoning_tokens - b.reasoning_tokens,
        cached_tokens=a.cached_tokens - b.cached_tokens,
    )
