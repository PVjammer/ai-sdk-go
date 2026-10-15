# Compaction v2 — porting baish's context machinery onto existing hooks

## Context

Survey done 2026-10-14 (`~/Workspace/notes/agent-loop-survey/`, not checked into this repo) compared
baish's hand-rolled agent loop against this SDK's `AgentBase`/`executeAgentLoop` and against pi
(`earendil-works/pi`). Finding that matters most: **baish built its own loop beside `pkg/agent`, not
on top of it.** baish has real, working compaction; `pkg/agent` has the hook skeleton (`PreModel`,
`PostTool`, `PostResponse`, etc., `base.go:256-268`) but nothing plugged into it yet. `agentcontext.Extended`
(`pkg/agentcontext/extended.go`) is a third, disconnected piece — a working, pluggable
`CompressionHook` framework with its own message store — that `AgentBase` doesn't use at all; it keeps
its own `agentSession.history []llm.ChatMessage` directly.

This doc is about consolidating these three into one thing, with baish's real mechanisms and pi's two
concrete improvements, not building a fourth parallel system.

## Decisions

"`AgentBase` adopts `agentcontext.Extended` as its session store" — currently `agentSession.history`
is a bare slice; `Extended` already has the right shape (per-context message store, pluggable
compression, token-threshold triggering) and nothing but wiring stands between them. Latitude on how
the migration is sequenced (adapter first vs. direct replacement) is left to whoever implements it.

estimateTokensForMessages (`agentcontext/extended.go:305-326`) and baish's `estimateTokens`
(`loop.go:638-650`) are the **same bug in two places**: `len(content)/4`. Both get replaced by one
shared helper:

```go
// TokenCounter abstracts "how many tokens does this cost," preferring the
// provider's own accounting over any client-side estimate.
abstraction TokenCounter
  func CountFromUsage(usage llm.Usage) int     # usage.TotalTokens, falling back to
                                                 # summed input+output+cache components
  func EstimateFromText(s string) int           # char/4 fallback, used ONLY when no
                                                 # usage object is available at all
                                                 # (e.g. counting a not-yet-sent draft)
```

Direct port of pi's `calculateContextTokens`/`getLastAssistantUsage` (`compaction.ts:140-175`):
walk history backward for the last assistant message with valid (non-aborted, non-zero) usage, use
its reported total. `EstimateFromText` only covers the gap where nothing has been sent to the
provider yet.

### Typed compaction output, not prose

Direct port of pi's `CompactionResult<T>` (`compaction.ts:104-114`) and its accumulating
`CompactionDetails{readFiles, modifiedFiles}` (`compaction.ts:51-88`), replacing baish's
`**Relevant Files:**` prose bullet (`compaction.go:83-96`):

```go
CompactionResult
  summary string
  firstKeptEntryID string
  tokensBefore int
  estimatedTokensAfter int
  usage llm.Usage
  details CompactionDetails         # extension point, see below

CompactionDetails
  readFiles []string                 # merged across rounds: previous CompactionDetails'
  modifiedFiles []string             # lists + newly observed tool calls this round
  "whatever else a particular agent's tool set wants tracked as structured data
   instead of re-parsed prose — this struct is meant to grow"
```

extractFileOperations-equivalent: every compaction round merges the PREVIOUS compaction's
`CompactionDetails` with newly-observed tool calls since then, so file-touched tracking survives
arbitrarily many rounds as data (`compaction.ts:60-88` is the reference implementation to port, not
reinvent).

### Invariants carried over unchanged from baish (not weakened, not "improved")

- **Turn-boundary-aware**: never compact the in-flight turn. baish's `l.turnStart` anchor
  (`loop.go:313`, `compaction.go:23`) is a hard requirement here too, not a nice-to-have — a compactor
  that doesn't honor this is wrong, full stop.
- **Update, not append**: re-compacting an already-compacted span edits the existing summary (moves
  completed Next-Steps into Progress, `compaction.go:99-107`) rather than concatenating a new one.
  Bounded summary growth across arbitrarily many compactions in one long session is the whole point;
  losing this is a regression even if the new summary format is otherwise nicer.
- **Non-destructive on failure**: if the compaction LLM call errors, history is untouched
  (`compaction.go`, implicit in baish's control flow — make it explicit here).

### Goal-pinning

Direct port of baish's `loop.go:566-578`: repeat the current turn's user message (truncated) near
the END of the system prompt on every round of that turn, not just at the start — exploits the
U-shaped attention curve, doesn't rely on it surviving history-trimming. Lands as a `PreModel` hook.
Whether this is still load-bearing once turn-branching (see `turn-branching.md`) keeps the trunk's
per-turn footprint small is an open question, not resolved here — keep it for the branch-internal
loop regardless, where a long multi-round turn can still drown the original ask.

### Repeated-call-signature nudge

Direct port of baish's stuck-loop detector (`loop.go:409-417`): same (tool name, args) signature
repeated ≥3× in one turn → nudge the model to stop looping, rather than let it spin until
`maxIterations`. Small, cheap, no reason not to have it.

## Deferred, post-turn tool-result compaction (the user's idea, 2026-10-14)

Distinct trigger point from everything above — not "context is under pressure" (reactive,
whole-history), but "this turn is answered, does this turn's own tool output still need to be
verbatim." Three trigger points now exist and are complementary, not redundant:

1. **Too big to inject at all** → `hooks.LargeResult` (`pkg/agent/hooks/large_result.go`), already
   exists, fires at tool-result time, redirects to a temp file.
2. **Turn answered, result no longer needed verbatim** → new: fires on `PostResponse`, walks that
   turn's tool results, and for anything over threshold that (1) didn't already trigger, prefers the
   same file-redirect shape over an LLM-summary call — no information loss, no extra LLM round trip.
   Falls back to LLM summarization only for results that don't compress well as files (conversational
   text, not data dumps).
3. **Whole session under context pressure** → the compaction described above, reactive,
   whole-history, the one real information-lossy step in this list.

This resolves `working-memory.md`'s open question #3 ("is phase-boundary flushing ergonomic, or does
it cover most real cases") in favor of: **turn/`PostResponse` boundary is the flush point**, using
`working-memory.md`'s own `wm:<uuid>` ref + `PreModel`-reinjection design as the mechanism, not a
new one. Concretely: a tool result goes into working memory (not raw history) at tool-result time;
`PostResponse` is the point where, if nothing in the rest of *this* turn needed the full content
re-injected, the ref is left standing permanently instead of re-expanded next turn. No separate
"compaction" step needed for this — it's just "stop reinjecting," which is strictly cheaper than
summarizing or truncating after the fact.

## What this does NOT change

`pkg/planner`'s `SimplePlanner`/`ReActPlanner` (text-pattern tool-call parsing, model-agnostic) are a
separate, parallel mechanism from `AgentBase`'s native-tool-calling path and aren't touched by this
doc — confirmed by reading, not assumed: `AgentBase` only ever calls `provider.ChatWithTools`
(`base.go:341`), never a planner. Whether these should eventually be reconciled (e.g. a planner as a
fallback `TokenCounter`-style abstraction for providers without native tool calling, mirroring how
baish's `ParseTextToolCalls` is a belt-and-suspenders net independent of whatever the provider already
tried) is a real question, flagged here, not answered.

## Relationship to other design docs

- `working-memory.md` — this doc resolves its open question #3 and names the concrete flush point;
  the `wm:` ref mechanism itself is unchanged from that doc's proposal.
- `session-dispatch.md` — unaffected; that doc's `Session`/steering/snapshot design is additive and
  orthogonal to compaction. Steering interacts with goal-pinning only in that a steer message injected
  mid-turn becomes part of what's pinned for the rest of that turn — worth a test case, not a design
  change.
- `turn-branching.md` — compaction becomes a **per-spine** concern once branching lands (every
  sufficiently turn-heavy branch needs its own compaction eventually, not just one privileged trunk);
  this doc's mechanisms are written to be spine-agnostic already (they operate on "a session's
  history," not on a hardcoded notion of *the* session), so no rework should be needed when that lands
  — but confirm this once both exist, don't assume it from this prose alone.

## Open questions

- `??` Should `TokenCounter` live in `pkg/llm` (next to `Usage`) or `pkg/agentcontext` (next to
  `Extended`, which needs it just as much as the ported baish compactor does)? Leaning `pkg/llm` since
  it's provider-shaped, not agent-shaped, but not decided.
- `??` `EstimateFromText`'s char/4 ratio is carried over unchanged from both sides for now — pi's
  README didn't document a better client-side estimator either, it just avoids needing one by reading
  provider usage. Worth a real tokenizer-based estimate later, not blocking this round.
