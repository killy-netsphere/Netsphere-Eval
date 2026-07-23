# Netsphere-Eval

**A model-agnostic, judge-free benchmark harness for self-hosted LLMs — built to
discriminate between frontier-class models after they saturate public suites.**

Point it at any number of OpenAI-compatible endpoints (vLLM, llama.cpp server,
SGLang, TGI, hosted APIs). Stdlib Python only: no pip installs, no external
services, no judge model. Every gold answer is computed by code, every scorer has
a self-test, and model pairs are compared with exact McNemar statistics on
identical items — so a "win" is a claim you can defend, not a vibe.

```bash
# smoke-test the pipeline (no servers needed)
python3 harness.py run --mock \
  --model '{"name":"a","url":"http://x/v1","model":"a"}' \
  --model '{"name":"b","url":"http://x/v1","model":"b"}' \
  --out results/mock

# benchmark one endpoint now, compare against archived runs later
python3 harness.py run \
  --model '{"name":"my-model","url":"http://127.0.0.1:8000/v1","model":"my-model"}' \
  --out results/my-model --seed 7

python3 harness.py compare --runs results/my-model results/other-model --out results/cmp
```

## Why v2 exists

Our v1 suite (115 items: math, code, tool-schema, instruction-following,
trick questions, needle retrieval) is included in `tasks/` and is fully
runnable — and **fully saturated**: three very different frontier-class models
(a 753B MoE at 3-bit, a 284B MoE at W4A16, and a day-one 118B MoE at FP8)
scored 90-96 of 99 paired items, all pairwise p > 0.2. A test everyone aces
carries no information.

v2 targets a 40-75% pass band with four categories:

| category | n | what it stresses | public tier |
|---|---|---|---|
| `math_hard` | 20 | multi-step chains where one slip cascades (CRT systems, exact rationals, determinants, recurrences, modular order) | **generator open** (`generators/gen_math_hard.py`) |
| `grounded_v2` | 30 | hallucination resistance: entirely fictional passages, synthesis questions, plausible-but-absent traps, half-answerable partials | **withheld** (see integrity policy) |
| `deepctx` | 15 | 128K-token ledgers: 5-hop chains, superseded records with as-of dates, near-miss decoys, absent needles; accuracy reported per needle depth | **generator open** (`tasks/gen_deepctx.py`) |
| `agentic` | 15 | real multi-turn tool loops: goal-stated prompts, error recovery, unannounced stale-data traps, a decoy tool, an IMPOSSIBLE task; scored on answer **and** call trace | **runner open** (`agent_runner.py`), task set withheld |

In our first three-way run, v2 produced two statistically significant quality
separations that v1 could not see — including one model fabricating facts on
7 trap items where another refused all 30, exact two-sided McNemar p = 0.0156.
Full results: [`docs/RESULTS-2026-07.md`](docs/RESULTS-2026-07.md).

## The integrity policy (why some sets are withheld)

**Publishing an exam burns the exam.** Static probe items that appear in a
public repo eventually appear in training corpora, and a benchmark that can be
memorized measures memory, not capability. So Netsphere-Eval is split into two
tiers:

- **The instrument is open.** The harness, scorers, statistics, the multi-turn
  agent runner with its trace predicates, and the *generators* for `math_hard`
  and `deepctx`. Generated categories are contamination-resistant by
  construction: every seed yields new instances whose answers must be computed
  from the instance itself — knowing the generator does not solve the item.
  Run any fresh seed and your numbers are honest:

  ```bash
  python3 - <<'EOF'
  import json, importlib.util as ilu
  s = ilu.spec_from_file_location("g", "generators/gen_math_hard.py")
  m = ilu.module_from_spec(s); s.loader.exec_module(m)
  with open("tasks/math_hard.jsonl", "w") as f:
      for it in m.gen(seed=YOUR_SEED, n=20):
          f.write(json.dumps(it) + "\n")
  EOF
  python3 harness.py run --only math_hard,deepctx --seed YOUR_SEED \
    --model '{"name":"m","url":"http://127.0.0.1:8000/v1","model":"m"}' --out results/m
  ```

- **The exam papers are withheld.** The 30 `grounded_v2` items (five fictional
  passages and their trap structure) and the 15 `agentic` task specs are static
  and memorizable, so they stay in our private registry. Fresh-seeded example
  items for the generated categories ship in `tasks/*.example.jsonl`; the
  grounded and agentic categories are documented by structure in
  [`docs/METHOD.md`](docs/METHOD.md) with invented illustrations, never real
  items. Our archived baselines were run before anything was published, and
  example items shown anywhere are flagged as burned in our registry.

If you want to *compare against our baselines* rather than run your own seed,
that requires the private item set — open an issue and we may run your
endpoint against the registry ourselves.

## How the items were built (and broken)

Every v2 item survived an adversarial pipeline before any benchmarked model saw
it: independent drafting agents wrote generators and items; separate verifier
agents then attacked each component on three lenses — *are the golds actually
correct* (recomputed by different methods), *would a correct answer be scored
wrong* (unit suffixes, format variants, tolerance), and *is it actually hard* —
followed by a fix round and a second, independent re-verification. Two defects
that survived the first pass and were caught by the second: a numeric-tolerance
hole that scored canonically-wrong continued fractions as correct on 48/91
instances, and a near-miss screen that silently made one item family
unsatisfiable. Both scoring sins — correct-rejected and wrong-accepted — were
probed to zero against the live scorer before the first benchmark run.

Scoring is objective everywhere: exact/numeric match with an accepted-variants
list, executed unit tests for code, programmatic constraint checks, and trace
predicates for agent runs (`called`, `not_called`, `min_calls`, `order`,
`recovered_from_error`, ...). Thinking output is stripped before scoring;
truncation counts as failure; reasoning-token budgets auto-scale from each
model's thinking knobs so heavy reasoners aren't strangled and light ones
aren't padded.

## Safety

The v1 `code` category **executes model-generated Python** (`python -I`,
timeout, temp dir — no syscall sandbox). Run the harness in a container or as a
throwaway user, never privileged. The v2 public categories execute nothing.

## Provenance

Built and used in production by **NetSphere**, a sovereign self-hosted AI
platform, to make model-selection decisions on real hardware. The results in
`docs/RESULTS-2026-07.md` were measured on a 4x 96 GB Blackwell workstation
node in a single overnight session, at temp 0, concurrency 1, identical items
and seeds across models.

MIT license. Trust is measured, not assumed.
