"""Offline tests for the Kimi client.

No network: a `FakeTransport` replaces the urllib opener, and `kimi._sleep` is
monkeypatched so the backoff path is exercised at full speed. Everything here
asserts against behaviour measured on the live endpoint -- the truncation
path and the cached-token price in particular.
"""

import email.message
import io
import json
import urllib.error
import urllib.request

import pytest

from autobranch import kimi as kimi_mod
from autobranch.kimi import (
    CONTEXT_WINDOW,
    Kimi,
    KimiError,
    KimiRateLimited,
    KimiTruncated,
    Usage,
    extract_json_object,
)

TOKENS = {"token_id": "ti", "token_secret": "ts"}


# ---- fakes --------------------------------------------------------------


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


class FakeTransport:
    """Scripted stand-in for `urllib.request.build_opener()`.

    Each scripted item is either a response dict (serialised to JSON) or an
    exception instance to raise. The last item repeats, so a test that only
    cares about the happy path can script one response and call twice.
    """

    def __init__(self, *script: object) -> None:
        self.script = list(script)
        self.requests: list[urllib.request.Request] = []
        self.timeouts: list[float | None] = []

    def open(self, request, timeout=None):  # noqa: ANN001 - mirrors urllib
        self.requests.append(request)
        self.timeouts.append(timeout)
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, BaseException):
            raise item
        return FakeResponse(json.dumps(item).encode())

    @property
    def bodies(self) -> list[dict]:
        """The JSON payload of each request, in order."""
        return [json.loads(r.data.decode()) for r in self.requests]


def completion(
    content: str = "ok",
    *,
    reasoning: str = "",
    finish_reason: str = "stop",
    prompt: int = 100,
    completion_tokens: int = 20,
    reasoning_tokens: int = 0,
    cached: int = 0,
) -> dict:
    return {
        "id": "cmpl-1",
        "model": kimi_mod.DEFAULT_MODEL,
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "reasoning_content": reasoning,
                },
            }
        ],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt + completion_tokens,
            "prompt_tokens_details": {"cached_tokens": cached},
            "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
        },
    }


def http_error(code: int, body: str = "nope", *, retry_after: str | None = None):
    headers = email.message.Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError(
        "https://example.invalid/v1/chat/completions",
        code,
        "boom",
        headers,
        io.BytesIO(body.encode()),
    )


def client(*script: object, **kw) -> tuple[Kimi, FakeTransport]:
    transport = FakeTransport(*(script or (completion(),)))
    return Kimi(opener=transport, **{**TOKENS, **kw}), transport


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Record backoff instead of serving it."""
    slept: list[float] = []
    monkeypatch.setattr(kimi_mod, "_sleep", slept.append)
    return slept


MSG = [{"role": "user", "content": "hi"}]


# ---- cost arithmetic ----------------------------------------------------


def test_cost_uses_published_rates():
    usage = Usage(prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert usage.cost_usd == pytest.approx(12.00)


def test_cached_prompt_tokens_are_discounted_and_not_double_counted():
    # Cached tokens are a SUBSET of prompt_tokens: half cached costs
    # 0.5 * $2.00 + 0.5 * $0.20, not $2.00 + $0.10.
    usage = Usage(prompt_tokens=1_000_000, cached_tokens=500_000)
    assert usage.cost_usd == pytest.approx(1.10)
    assert Usage(prompt_tokens=1_000_000, cached_tokens=1_000_000).cost_usd == pytest.approx(0.20)


def test_reasoning_tokens_bill_as_output_inside_completion_tokens():
    usage = Usage(prompt_tokens=0, completion_tokens=143, reasoning_tokens=102)
    assert usage.cost_usd == pytest.approx(143 * 10.0 / 1_000_000)
    assert usage.reasoning_fraction == pytest.approx(102 / 143)


def test_usage_sums_and_zero_is_identity():
    a = Usage(1, 2, 3, 4)
    assert a + Usage() == a
    assert a + a == Usage(2, 4, 6, 8)
    assert Usage().reasoning_fraction == 0.0


def test_usage_parse_tolerates_missing_detail_blocks():
    assert Usage.parse({"prompt_tokens": 7, "completion_tokens": 3}) == Usage(7, 3, 0, 0)
    assert Usage.parse(None) == Usage()


# ---- happy path ---------------------------------------------------------


def test_chat_separates_content_from_reasoning_and_bills_the_client():
    k, transport = client(
        completion(
            "the answer", reasoning="thinking hard", completion_tokens=50,
            reasoning_tokens=40, cached=64,
        )
    )
    reply = k.chat(MSG, max_tokens=1000)
    assert reply.content == "the answer"
    assert reply.reasoning == "thinking hard"
    assert reply.finish_reason == "stop"
    assert reply.usage.reasoning_tokens == 40
    assert reply.usage.cached_tokens == 64
    assert k.calls == 1
    assert k.total_usage == reply.usage
    assert k.total_usage.cost_usd > 0

    body = transport.bodies[0]
    assert body["model"] == kimi_mod.DEFAULT_MODEL
    assert body["max_tokens"] == 1000
    assert body["messages"] == MSG
    assert "response_format" not in body
    assert transport.timeouts[0] == k.timeout


def test_json_object_sets_response_format():
    k, transport = client(completion('{"a": 1}'))
    k.chat(MSG, json_object=True)
    assert transport.bodies[0]["response_format"] == {"type": "json_object"}


def test_default_max_tokens_is_used_when_unspecified():
    k, transport = client(default_max_tokens=777)
    k.chat(MSG)
    assert transport.bodies[0]["max_tokens"] == 777


def test_missing_credentials_fails_at_construction_naming_the_secret(monkeypatch):
    monkeypatch.delenv("MODAL_PROXY_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_PROXY_TOKEN_SECRET", raising=False)
    with pytest.raises(KimiError, match="autobranch-kimi-auth"):
        Kimi()


def test_env_vars_supply_defaults(monkeypatch):
    monkeypatch.setenv("MODAL_PROXY_TOKEN_ID", "env-id")
    monkeypatch.setenv("MODAL_PROXY_TOKEN_SECRET", "env-secret")
    monkeypatch.setenv("AUTOBRANCH_KIMI_URL", "https://custom.example/")
    k = Kimi(opener=FakeTransport(completion()))
    assert (k.token_id, k.token_secret) == ("env-id", "env-secret")
    assert k.base_url == "https://custom.example"  # trailing slash stripped


# ---- affinity key -------------------------------------------------------


def test_affinity_key_is_sent_on_every_request_even_when_unset():
    k, transport = client(completion(), completion())
    k.chat(MSG)
    k.chat(MSG)
    keys = [r.get_header("Modal-routing-affinity-key") for r in transport.requests]
    assert keys == [k.affinity_key, k.affinity_key]
    assert keys[0]  # a client with no key still pins itself to one worker


def test_affinity_key_per_call_overrides_the_client_default():
    k, transport = client(completion(), completion(), affinity_key="branch-root")
    k.chat(MSG)
    k.chat(MSG, affinity_key="branch-a1")
    assert [r.get_header("Modal-routing-affinity-key") for r in transport.requests] == [
        "branch-root",
        "branch-a1",
    ]


def test_auth_headers_are_the_modal_proxy_pair():
    k, transport = client()
    k.chat(MSG)
    request = transport.requests[0]
    assert request.get_header("Modal-key") == "ti"
    assert request.get_header("Modal-secret") == "ts"


# ---- the truncation trap ------------------------------------------------


def test_empty_content_with_finish_length_retries_once_at_double_budget():
    truncated = completion(
        "", reasoning="let me think about the reward weights",
        finish_reason="length", completion_tokens=16, reasoning_tokens=16,
    )
    k, transport = client(truncated, completion("finally an answer"))
    reply = k.chat(MSG, max_tokens=16)
    assert reply.content == "finally an answer"
    assert [b["max_tokens"] for b in transport.bodies] == [16, 32]
    # Both calls were billed: a truncated call still burned output tokens.
    assert k.calls == 2
    assert k.total_usage.completion_tokens == 36


def test_truncation_twice_surfaces_a_distinct_retryable_error():
    truncated = completion(
        "", reasoning="still thinking", finish_reason="length",
        completion_tokens=16, reasoning_tokens=16,
    )
    k, transport = client(truncated)
    with pytest.raises(KimiTruncated) as caught:
        k.chat(MSG, max_tokens=64)
    assert "never stopped thinking" in str(caught.value)
    assert isinstance(caught.value, KimiError)
    assert [b["max_tokens"] for b in transport.bodies] == [64, 128]


def test_truncation_is_not_retried_past_the_context_window():
    truncated = completion("", finish_reason="length", completion_tokens=5, reasoning_tokens=5)
    k, transport = client(truncated)
    with pytest.raises(KimiTruncated):
        k.chat(MSG, max_tokens=CONTEXT_WINDOW)
    assert len(transport.requests) == 1


def test_empty_content_with_reasoning_but_finish_stop_is_also_truncation():
    # Observed shape: the model burns its turn on reasoning_content and stops
    # without answering. Doubling the budget is still the fix.
    k, _ = client(
        completion("", reasoning="hmm", finish_reason="stop", reasoning_tokens=20),
        completion("answer"),
    )
    assert k.chat(MSG).content == "answer"


def test_genuinely_empty_answer_without_reasoning_is_not_truncation():
    k, _ = client(completion("", finish_reason="stop"))
    assert k.chat(MSG).content == ""


# ---- retries ------------------------------------------------------------


def test_429_is_retried_with_growing_jittered_backoff(no_sleep):
    k, transport = client(
        http_error(429, "rate limited"), http_error(429), completion("through")
    )
    assert k.chat(MSG).content == "through"
    assert len(transport.requests) == 3
    assert len(no_sleep) == 2
    assert all(0 < s <= kimi_mod._BACKOFF_CAP_SECONDS for s in no_sleep)
    # Jitter means we assert the envelope, not the value: attempt n sleeps
    # within [0.5, 1.0] x the 0.5 * 2**n ceiling.
    assert no_sleep[0] <= 0.5
    assert no_sleep[1] <= 1.0


def test_429_honours_retry_after_when_present(no_sleep):
    k, _ = client(http_error(429, retry_after="3.5"), completion())
    k.chat(MSG)
    assert no_sleep == [3.5]


def test_429_past_max_retries_raises_rate_limited(no_sleep):
    k, transport = client(http_error(429, "still limited"), max_retries=2)
    with pytest.raises(KimiRateLimited, match="429"):
        k.chat(MSG)
    assert len(transport.requests) == 3  # initial + 2 retries
    assert len(no_sleep) == 2


def test_5xx_is_retried(no_sleep):
    k, transport = client(http_error(503, "overloaded"), completion("recovered"))
    assert k.chat(MSG).content == "recovered"
    assert len(transport.requests) == 2


def test_4xx_is_not_retried_and_echoes_the_body(no_sleep):
    k, transport = client(http_error(400, "bad max_tokens"))
    with pytest.raises(KimiError, match="bad max_tokens"):
        k.chat(MSG)
    assert len(transport.requests) == 1
    assert no_sleep == []


def test_transport_errors_are_retried(no_sleep):
    k, transport = client(urllib.error.URLError("connection reset"), completion("ok"))
    assert k.chat(MSG).content == "ok"
    assert len(transport.requests) == 2


def test_non_json_body_is_a_clear_error():
    class Garbage(FakeTransport):
        def open(self, request, timeout=None):
            self.requests.append(request)
            return FakeResponse(b"<html>504 gateway</html>")

    k = Kimi(opener=Garbage(), **TOKENS)
    with pytest.raises(KimiError, match="non-JSON"):
        k.chat(MSG)


def test_endpoint_error_payload_is_surfaced():
    k, _ = client({"error": {"message": "model not loaded", "type": "server_error"}})
    with pytest.raises(KimiError, match="model not loaded"):
        k.chat(MSG)


# ---- JSON extraction ----------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        '{"tool": "evaluate"}',
        '```json\n{"tool": "evaluate"}\n```',
        '```\n{"tool": "evaluate"}\n```',
        'Here is the action:\n\n{"tool": "evaluate"}\n\nLet me know if that works.',
        'Sure.\n```json\n{"tool": "evaluate"}\n```\nI chose this because the objective matters.',
    ],
)
def test_chat_json_tolerates_fences_and_prose(body):
    k, _ = client(completion(body))
    assert k.chat_json(MSG) == {"tool": "evaluate"}


def test_extract_handles_nesting_and_braces_inside_strings():
    text = 'prose {"a": {"b": [1, 2]}, "note": "a } brace and a \\" quote"} trailing'
    parsed = json.loads(extract_json_object(text))
    assert parsed["a"]["b"] == [1, 2]
    assert parsed["note"] == 'a } brace and a " quote'


def test_extract_rejects_unbalanced_and_absent_objects():
    with pytest.raises(ValueError, match="no '{'"):
        extract_json_object("no object here")
    with pytest.raises(ValueError, match="never closes"):
        extract_json_object('{"a": 1')


def test_chat_json_error_echoes_a_truncated_body():
    long_prose = "I will not answer in JSON. " * 60
    k, _ = client(completion(long_prose))
    with pytest.raises(KimiError) as caught:
        k.chat_json(MSG)
    message = str(caught.value)
    assert "found none" in message
    assert "I will not answer in JSON." in message
    assert "chars]" in message  # echo was clipped rather than dumped whole


def test_chat_json_rejects_a_non_object_top_level():
    k, _ = client(completion("[1, 2, 3]"))
    with pytest.raises(KimiError, match="no '{'|expected a JSON object"):
        k.chat_json(MSG)


def test_chat_json_schema_hint_is_appended_without_mutating_the_caller_list():
    messages = list(MSG)
    k, transport = client(completion('{"ok": true}'))
    assert k.chat_json(messages, schema_hint='{"ok": bool}') == {"ok": True}
    assert messages == MSG  # caller's list untouched
    sent = transport.bodies[0]["messages"]
    assert len(sent) == 2
    assert '{"ok": bool}' in sent[-1]["content"]
    assert transport.bodies[0]["response_format"] == {"type": "json_object"}
