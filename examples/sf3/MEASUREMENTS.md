# SF3 run: what was measured before anything was built on it

Absolute values from real probes against the real model and the real ROM.
Written down because every one of these decided a design choice, and a
restarted session that re-derives them pays twice.

## The reproduction target — modal-labs/sf3, read from the source

| what | value | where |
|---|---|---|
| base model | `Qwen/Qwen3-VL-8B-Instruct` | `src/train/main.py` |
| fine-tune scope | **full parameter**, no LoRA anywhere | `modal-dojo==0.1.1` → slime/Megatron; `convert_megatron_checkpoint_to_hf`; `iter_*_hf` loaded as the whole model |
| GPUs | 8 per node | `actor_num_gpus_per_node=8` |
| rollouts | 100 | `NUM_ROLLOUTS` |
| character | **sampled per fight** from the 19-name roster | `random.sample(ROSTER, 2)` |
| super art | 1 | `OUTFIT = SUPER_ART = 1` |
| opponent | arcade CPU, PFSP difficulty sampling 1..8 | `_sample_cpu_difficulty` |
| his reward | `outcome*1.0 + 0.1*clip(damage_return,-1,1) - (20/160)*invalid`, discounted on a 3600-frame half-life | `rollout.py:30-33, 207-226` |
| action format | free text, parsed, fallback on invalid | `MODEL.parse_response(...)` |

**Character choice here: Ryu.** His trainer samples randomly and his eval runs
every matchup, so there is no single character to copy. Ryu is in his roster,
is what the predecessor rl-hackathon run used (`sf/actions.py:19`), and so
makes this run's number directly comparable to that project's measured
`r2_hinh` 55.0% and `r3_potential` 79.0% at level 8.

## Probe 1 — can an 8.77B VL model take a full-parameter update?

`probe_full_ft.py`, H100, real 384x224 frame, `bnb.optim.AdamW8bit`,
gradient checkpointing.

```
params                8.77 B, all trainable      (16.3 GiB bf16)
peak allocated        49.4 GiB
peak reserved         49.5 GiB  of 79.2 available
one fwd+bwd+step      2.93 s   (batch of 1)
prompt tokens         104      (one frame + the state line)
fits on one 80 GB     yes
```

**Decided:** one H100 per branch, no model parallelism. The arithmetic said
fp32 Adam moments alone would be 64 GiB — over the card — so the 8-bit
optimizer is load-bearing, not a tuning preference. Hedging with `H100:2`
would have doubled the GPU bill for nothing.

**Also decided:** 104 tokens per decision means the frame is nearly free in
context, so there is no reason to downscale it or strip the HUD.

## Probe 2 — is an emulator lane cheap enough to be an environment?

`probe_sf3_env.py`, one Modal Sandbox, difficulty 8, 12 real macro actions.

```
boot                  27.3 s   (40.3 s wall, including sandbox create)
frame                 (224, 384, 3) uint8
step latency          mean 0.25 s, min 0.19, max 0.39  (n=12)
game frames/decision  6.7      (80 frames over 12 decisions)
state at reset        health 1.00  opp 1.00  timer 0.99  side 1
signals live          super_gain 0.0547 after a Hadouken
```

**Decided:** `max_decisions_per_match = 600` (~67 s of game time, about one
round, so a round *outcome* usually lands inside the collection window) and
`groups_per_iteration = 1` (a collection costs ~150 s of wall clock regardless
of lane count, because lanes run concurrently).

**Also found:** MAME fails with `libGL.so.1: cannot open shared object file`
unless `SDL_VIDEODRIVER=dummy` is set. Installing the SDL runtime libraries
looks like the fix and is not; the emulator image copies the predecessor
project's working package list and three environment variables verbatim.

## Throughput budget this implies

```
collection   600 decisions x 0.25 s                    = ~2.5 min
update       256 subsampled / 8 per micro-batch x 3.5 s = ~1.9 min
                                                  total  ~4.4 min / iteration
4 hours per branch                                       ~54 iterations
```

Comparable to his 100 rollouts. `micro_batch_size = 8` rather than 1 because
the probe left 30 GiB of headroom and a batch of one would spend 78 minutes
updating on a single 4800-decision collection.

## Framework defects this exposed, and the fixes

1. **`TrainResult` field names.** `grpo_full` first returned
   `steps=/checkpoint=`; the dataclass wants
   `steps_done/checkpoint_path/wall_seconds/iterations/workspace/reward_fingerprint`.
   `branch_main` reads them with `getattr(..., default)`, so the mismatch
   would not have raised — it would have silently recorded no checkpoint and
   published no weights, which is the "no artifact" failure this project has
   already paid for once.
2. **`RewardConfig` is `dict[str, Any]`, not a class.** `isinstance` against
   it raises at runtime and it has no `.parse`.
3. **The compiled reward is vectorised.** Calling it per lane costs N times as
   much and gives stateful shaping terms (potential, delta) a different
   history per lane than the trainer intends.
4. **`/workspace` is sandbox-local.** The 16 GiB HF export and `metrics.jsonl`
   would both have died with each branch — losing the trained model and every
   number needed for post-hoc analysis. Both now go to
   `/data/runs/<run>/<branch>/`, mirrored for metrics so a fork still
   inherits its own copy. Keeping 16 GiB out of `/workspace` also matters
   because every fork snapshots that directory.
