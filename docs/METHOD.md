# Method

## Design rules (inviolable)

1. **No judge model.** Every item is scorable by code: exact/numeric match with
   an accepted-variants list (`answer_any`), executed unit tests, programmatic
   constraint checks, or trace predicates. A benchmark judged by an LLM
   inherits that LLM's biases and drifts with its versions.
2. **Computed golds.** Wherever a computation exists, the gold is produced by
   code at generation time — never hand-asserted. Generators carry independent
   cross-checkers that recompute every gold by a different method.
3. **Deterministic from seed.** Same seed → byte-identical items, so paired
   statistics across models and time are valid.
4. **Both scoring sins probed to zero.** Before any model runs: (a) a battery
   of correct-but-reformatted answers must all PASS (unit suffixes, thousands
   separators, fraction forms, non-padded dates, orphaned reasoning tags);
   (b) a battery of canonical *wrong* answers — one-step-short evaluations,
   dropped terms, off-by-one numerators, stale-record values, wrong-unit
   labels — must all FAIL, including against numeric tolerance.

## Category anatomy

### math_hard (generator public)
Families where a single arithmetic slip cascades: simultaneous congruence
systems with coefficient inverses, exact rational chains (continued fractions,
telescoping with non-textbook splits, Möbius iterations), 5x5/6x6 integer
determinants, linear recurrences at depth under a modulus, multiplicative
order via CRT decomposition, inclusion-exclusion counts. Instances whose
plausible near-misses fall within 10x the scorer's numeric tolerance of the
gold are rejected and regenerated at build time, so the tolerance can never
bridge a wrong value to a pass. Every prompt states its exact answer format.

### grounded_v2 (withheld)
Passages of entirely fictional technical/bureaucratic content (invented
protocols, organisms, ordinances) so training data cannot help; the passage is
the only source of truth. Three item classes: **synthesis** (combine stated
facts, small computations — pure extraction is capped under 30% of the set),
**traps** (the passage plausibly *should* contain the asked fact but does not;
the only correct answer is exactly `NOT IN CONTEXT`; targets are chosen so a
"zero/none" reading is never defensible), and **partials** (half the question
is answerable, half is not; the item states an exact two-field reply template).
Invented illustration of a trap (not a real item):

> *Passage states a permit costs 40 florins, is valid 90 days, and renewal
> requires Form K.* Question: "What is the late-renewal penalty?" — The
> passage never states one. Gold: `NOT IN CONTEXT`. A model answering "40
> florins" or inventing a penalty is scored wrong; so is "no penalty" — the
> passage doesn't say that either, which is the point.

### deepctx (generator public)
A generated personnel/task ledger sized to a target real-token count
(calibrate chars/token against *your* tokenizer — ID-dense text runs ~2.2-2.4).
Item types: 5-hop reference chains where at least two hops pass through
entities with superseded records (the stated tie-break is the `as_of` date, and
following a stale record provably yields a *different* final answer — asserted
at generation); aggregations whose membership condition requires its own
lookup; near-miss decoys (`BLD-SABLE-8` vs `BLD-SABEL-8`); absent needles
where the correct answer is `NOT IN CONTEXT`. Needle positions are recorded
(10/50/90% depth) so reports break accuracy down by position.

### agentic (runner public, tasks withheld)
The harness plays the tool runtime: the model receives OpenAI tool schemas,
the harness executes deterministic pure-Python implementations, appends
`role:"tool"` results, and loops (turn/call budgets enforced). Prompts state
*goals*, never procedures. Failure modes engineered in: a tool that errors on
the natural first call (recovery required), two sources where one is quietly
stale — discoverable only via result metadata, a decoy tool that must not be
called, tasks with a genuinely impossible request where the only correct
answer is `IMPOSSIBLE`. Scoring requires the correct final answer AND a valid
trace (`called` / `not_called` / `min_calls` / `order` /
`recovered_from_error` — see `agent_runner.py`); a right answer obtained from
the stale source fails. Trace floors are audited against the logically minimal
correct strategy so efficient solvers are never penalized.

## Adversarial build pipeline

Draft (independent agents per component) → verify (three lenses per component,
by separate agents: gold correctness recomputed differently; would-a-correct-
answer-be-rejected; is-it-actually-hard) → fix → **independent re-verify that
does not trust the fix report and re-runs its own probes**. Components ship
only at zero fatal / zero major. Two second-round catches that justify the
pipeline: a numeric-tolerance hole scoring canonically-wrong continued
fractions as correct on 48/91 instances, and an over-eager near-miss screen
that silently extinguished an item family (structurally unsatisfiable filter).

## Statistics and reporting

Identical items, same seed, temp 0, concurrency 1. Exact two-sided McNemar on
discordant pairs per category and overall; ~80 items resolves ~12pp gaps —
high p is not equivalence. Reported per model: pass counts per category,
median completion tok/s, wall time; deepctx additionally by needle depth.
Truncation (`finish=length`) is failure by design. Reasoning budgets derive
from each model's declared thinking knobs (a max-effort knob gets 32,768; a
boolean thinking flag gets 16,384); budget-sensitivity claims are settled with
a labeled sensitivity arm, never by silently re-scoring.
