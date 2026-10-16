# Turn-scratch branching: evaluation

Status: evaluated against real models, real tasks, with replication. Not a design doc — see
`turn-branching.md` and `compaction-v2.md` for the mechanism itself. This is what happened when
it was actually run.

## Summary

Turn-scratch branching does what it was built for: it keeps live context small without losing
any information, by hiding a turn's raw tool-call chatter behind a short reference and
retrieving it on demand instead of carrying it forever. That part is now well-evidenced, not
just argued. The more interesting result is on **token cost**, which turned out to favor
branching far more than expected: in 5 of 6 model/workload combinations tested, branching used
*fewer* total tokens than the flat alternative, sometimes dramatically (6.2x fewer in one case) —
because a flat session's cost grows worse than linearly once it gets long or messy, and
branching's per-turn scratch sessions are immune to that. The cost branching does pay is real
but different from what you'd guess: more tool-call round trips, a fixed system-prompt tax on
every single call, and occasional extra "rediscovery" work when something needed later wasn't
retained — not runaway token growth.

The biggest open risk by far is not branching at all: on real multi-file coding work, only the
largest model tested (`qwen3.8-96k`, 27B) reliably produced working code. Two smaller models
(`gemma4-131k`, 7.5B; `llama32-64k`, 3B) failed to compile the deliverable in every single case,
**regardless of branching mode**. If you're evaluating this for production use, model capability
on the actual task is the gating factor, not this feature.

## What was tested

Two suites, run with `cmd/experiment-runner` (see `experiments/README.md`), each as 3 models x
2 branching modes x 2 replicates = 12 cases per suite:

- **`repo-feature.yaml`** — a cold-start, multi-turn coding task against this repo itself (via a
  fresh git worktree per case): explore an unfamiliar package's conventions, implement a new
  hook matching them, plus a deliberately natural "wrong directory, correct filename" discovery
  probe (ask for an edit to a file that doesn't exist at the given path; ask for a *second* edit
  to "that same file" 10 turns later, with no path restated, to see whether the correct location
  is retained or has to be rediscovered — or regresses to the wrong path again).
- **`daily-driver.yaml`** — a lightweight, quick-lookup "personal assistant" workload: doc
  search, a simulated web search, notes, time — many short single-tool-call turns instead of
  few heavy ones.

Models: `qwen3.8-96k` (dense, 27B), `gemma4-131k` (7.5B), `llama32-64k` (3B) — all local Ollama
models, each with a custom `num_ctx` baked in matching its actually-verified served context
(not the architecture's advertised max, which can be very different — see each model's
Modelfile derivation notes in the broader session history this evaluation came from).

Recall probes are ordinary follow-ups ("what did reminder.md say again?"), not "without doing X
again" instructions — an earlier version of this evaluation used explicit prohibitions, which
turned out to measure compliance with an unusual directive rather than real memory/retrieval
behavior. Dropped in favor of just observing what a model does naturally.

## Bugs found during validation (all fixed, see commit history)

Running this against real models surfaced several real bugs that unit tests alone hadn't
caught:

1. **`Session.leaf` data race** — unguarded field, now mutex-protected.
2. **Ollama's OpenAI-compat endpoint names the reasoning field `reasoning`, not the
   DeepSeek-style `reasoning_content`** this code only originally checked — a response
   truncated by the token limit (common for a reasoning model mid-`<think>`) came back with
   real content under a field this code didn't look at, surfacing as a hard "LLM returned
   empty response" error. Fixed in `pkg/llm/openai.go`, with a regression test.
3. **Compaction's trigger math was a fraction of a fraction** — `Threshold x (contextWindow x
   fraction)` — which compacted real sessions at under 10% of a model's actual context room.
   Replaced with pi's actual algorithm (trigger within a fixed `--compaction-reserve` of the
   real context window).
4. **The real root cause, found by code review, not by the original bug report**: `SeedSession`
   (called by Compaction's `PreModel` hook mid-turn) used to allocate a brand-new session
   object and swap it into the session map — silently detaching the `*agentSession` pointer
   `executeAgentLoop` already held for the rest of that turn. Every tool call and the final
   answer appended afterward went to an object nothing could read back. Fixed by mutating an
   existing session's history in place instead. Confirmed with a test that fails on the old
   code and passes on the fix.
5. Compaction re-invoked its summarizer every round even when there was nothing new to fold in
   — wasted, and the actual mechanism that let a stuck turn burn through its iteration budget
   faster than the stuck loop alone would have.

None of these are branching-specific risks in the sense of "branching is fragile" — they're the
kind of bug that live-model testing finds and unit tests don't, in any sufficiently stateful
system. Listed here because they directly affected earlier (now superseded) results in this same
evaluation.

## Results

### Context size: branching wins consistently

Across every model and both suites, branch-on's live context stayed a small fraction of
branch-off's — as low as ~8% on the real-repo coding task, higher (~30-60%) on the lighter
daily-driver workload, since savings scale with how much raw tool-call volume a turn generates
relative to its final answer. Never reversed, in any case tested.

### Token cost: branching usually wins here too, sometimes by a lot

This is the result that most needed a correction once actually measured — the raw "~2x the
tool calls" number looks alarming in isolation and does **not** mean ~2x the tokens.

| model | suite | branch-on tokens | branch-off tokens | ratio |
|---|---|---|---|---|
| gemma4-131k | repo-feature | 866,728 | 5,406,838 | **16%** |
| qwen3.8-96k | repo-feature | 350,640 | 997,900 | **35%** |
| llama32-64k | repo-feature | 102,272 | 84,233 | 121% (on costs *more*) |
| gemma4-131k | daily-driver | 41,422 | 74,331 | **56%** |
| llama32-64k | daily-driver | 37,538 | 50,515 | **74%** |
| qwen3.8-96k | daily-driver | 110,977 | 168,671 | **66%** |

Why the gemma repo-feature number is so extreme: that case's branch-off sessions degraded into
flailing (confused edits, long write/test loops), and context grew to 255K+ characters. Every
round under flat history re-transmits the *entire* accumulated context as prompt tokens — so a
session that grows messy doesn't cost more linearly, it costs more *quadratically*. Branching's
per-turn scratch sessions are structurally immune to that: each turn's cost is bounded by that
turn's own content, not the ever-growing trunk. This is the scenario branching's token advantage
compounds hardest in — a long, messy session is exactly where you need it most, not where it
breaks down.

The one reversal (llama32 on repo-feature) is informative in a different way: that case barely
did any real tool-driven work in either mode (it mostly failed the task outright — see below),
so there was no large context to save on, and what was left was a real, fixed cost: branch-on's
system prompt is longer (it has to explain `list_branches`/`read_branch`), and that overhead
applies to *every* round, used or not. Confirmed directly in an isolated 2-turn smoke test:
branch-on's very first turn cost more prompt tokens than branch-off's, before any context had
diverged between them at all. Branching's token advantage needs a long-enough or heavy-enough
session to pay for its own fixed overhead; on something that stays small throughout, it may not.

### Wall-clock time: branching costs more here, and that's a separate axis from tokens

Branch-on made roughly 2x the tool calls of branch-off for qwen on repo-feature, but that only
cost ~21% more wall-clock time — not 2x — because the extra calls are cheap retrieval/bookkeeping
operations (`read_branch`, `list_branches`, re-exploring via `list_files`), not expensive
generation calls. More tool calls means more round trips, and each round trip has fixed latency
overhead independent of how many tokens it carries — so branch-on can genuinely use *fewer*
tokens while still taking *longer*. If you're paying per-token, branching is close to uniformly a
win per the table above. If you're latency-sensitive running serially, the extra round trips are
a real, separate cost.

### Rediscovery cost (the wrong-path probe)

Quantified directly (qwen, the one model reliable enough to isolate the effect from its own
task failures): at the specific turn designed to test it, branch-on needed 2-4 extra tool calls
to either retrieve the original discovery or rediscover it from scratch — comparable to, not
dramatically worse than, branch-off's cost at the same turn. The bigger, more pervasive cost was
habitual: branching sessions reached for retrieval across *many* turns, not just the one
designed to need it, including one case re-deriving facts about a file it had already explored
twice earlier in the same session. Real, worth knowing, not catastrophic.

### Task reliability: the dominant variable, and it's not about branching

Checked by actually running `go test` against each case's resulting code, not just reading the
mechanical pass/fail report:

| model | cases producing working code (of 8, both suites combined) |
|---|---|
| qwen3.8-96k | 8/8 |
| gemma4-131k | 1/8 |
| llama32-64k | 0/8 |

This cuts identically across both branching modes — gemma's two branch-*off* replicates on the
wrong-path probe were the ones that went furthest off the rails (editing an unrelated file
entirely), not the branch-on ones. Model capability on the underlying task, not branching mode,
is what determines success or failure here.

## Open questions, not yet tested

- Spine-level compaction: the trunk itself has no compaction mechanism and grows forever over a
  long enough session (hundreds of turns). Untested at that scale.
- Discovery at a large branch count (20-30+): does `list_branches`' own catalog output become
  a new source of bloat?
- The "stub = reasoning/post-tool-response" idea (keep more of a turn's own narration on the
  trunk, not just the final answer, trading some context savings for fewer retrieval misses) —
  proposed, not implemented or tested.
- `daily-driver` was never pushed into the kind of failure mode `repo-feature` hit naturally;
  unclear whether a long enough light-workload session would eventually show the same
  quadratic-cost pattern under flat history.

## Rating

**Core mechanism: 8/10.** Context savings and token savings both held up under real replication,
across models, across two different workload shapes. Retrieval never returned wrong data —
when it needed extra work, it cost rounds, not correctness.

**Confidence this is production-ready as a default: 6/10.** The main residual risk isn't
branching's own behavior, which has now been measured in enough detail to trust at a given
scale — it's that this SDK's agent loop, independent of branching, isn't yet reliable enough on
real multi-step coding work for anything smaller than a 27B-class model. That's a separate
problem from the one this document evaluates, but it's the one that actually gates how far down
the model-size ladder any of this can go today.
