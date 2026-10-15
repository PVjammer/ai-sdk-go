# Roadmap — agent-loop-v2

Branch: `agent-loop-v2`. Three design docs feed this, each with its own internal ordering already —
this file merges them into one sequence, it doesn't invent a new one from scratch.

Phase numbers are dependency order, not sprints. Nothing here is scheduled against dates.

## Phase 0 — confirm before building on it

- `??` from `turn-branching.md`: whether pi's tree (`session-manager.ts`) is scoped per-session-file
  or shared globally across `/fork`s. Affects whether `SessionLog` is one-per-top-level-session or
  broader. Five-minute check, blocks Phase 4 specifically, blocks nothing else.

## Phase 1 — `session-dispatch.md`'s own order (unchanged, already decided in that doc)

1. `AccumulateEvents` / `RunResult` — low effort, high immediate value, no dependencies.
2. `Session` object (wraps `AgentBase`, owns one session's lifecycle).
3. Steering + follow-up queues (needs the `Session` object from #2).
4. Session snapshot/restore (flat JSON, needs `Session` object from #2).

Independent of everything below. Can land in parallel with Phase 2.

## Phase 2 — compaction consolidation (`compaction-v2.md`)

1. `TokenCounter` (`CountFromUsage` / `EstimateFromText`) — replaces the duplicated `len(content)/4`
   heuristic in both baish (reference only, not this repo) and `agentcontext/extended.go:305-326`.
2. `CompactionResult` / `CompactionDetails` typed output.
3. Port baish's compaction logic itself (turn-boundary-aware, update-not-append, non-destructive on
   failure) as an `agentcontext.CompressionHook` implementation — the hook interface already exists
   in `agentcontext`, this is new logic behind the existing interface, not a new interface.
4. Wire `AgentBase` to use `agentcontext.Extended` as its session store instead of its own flat
   `agentSession.history` — the real consolidation step; everything in this phase is inert until this
   lands, since `AgentBase` doesn't currently read from `Extended` at all.
5. Goal-pinning (`PreModel` hook) and repeated-call-signature nudge — small, independent, can land
   anywhere in this phase or Phase 3.

## Phase 3 — loop-shape fixes (`turn-branching.md`'s tool-concurrency section)

- Parallel tool-call execution by default, with per-tool sequential opt-out. Independent of Phase 2;
  touches `executeAgentLoop`'s tool-dispatch loop (`base.go:392-524`), not the compaction path.

## Phase 4 — deferred post-turn result compaction (`compaction-v2.md`'s "Deferred..." section)

- `working-memory.md`'s `wm:` ref mechanism, with the flush point resolved as `PostResponse`
  (see that doc's "Resolution (2026-10-15)" note).
- Depends on Phase 2 step 4 (needs `Extended` actually wired in) and the `PostResponse` hook path
  being exercised by something already (Phase 2 step 3 exercises it first).

## Phase 5 — `SessionLog` tree primitive (`turn-branching.md`)

- `Entry` / `SessionLog` (append, `branchFrom`, `getChildren`, `pathTo`). New code, no existing
  equivalent in this repo (confirmed — `session-dispatch.md` explicitly chose flat snapshots instead,
  for a different use case; see that section of `turn-branching.md` for why both are kept).
- Biggest, riskiest single piece of this roadmap. Depends on Phase 0.

## Phase 6 — turn-scratch branching (`turn-branching.md`)

- Retroactive branch-or-splice decision at `PostResponse`, tool-message branch reference, `read_branch`
  tool. Depends entirely on Phase 5.
- Compaction (Phase 2) must be re-verified as spine-generic once this lands — stated as a design
  requirement in both docs, not yet checked against real code.

## Explicitly not scheduled (flagged, not forgotten)

- Hermes's self-authored-skill pipeline (`/learn` → `skill_manage` → learning graph) — a genuinely
  different, bigger idea than anything else here. `pkg/skill` already exists (catalog XML in system
  prompt, `skillLoader`) as a starting point if this is ever picked up, but it is not part of this
  roadmap.
- AnythingLLM-style retrieval-augmented memory (embedded vector store per workspace) as an alternative
  or complement to compaction. Noted as a real axis worth weighing later, not adopted this round —
  `read_branch` (Phase 6) already gives a partial answer to "nothing is really gone," which covers
  some of the same motivation without a new subsystem.
