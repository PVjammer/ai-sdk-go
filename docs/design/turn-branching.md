# Turn branching — tree-structured session history

## Context

`session-dispatch.md` §4 explicitly rejected pi's tree-structured session log for this SDK:

> This intentionally does not implement pi's JSONL tree branching. Flat snapshots cover the daemon
> restart use case without the complexity of navigable history trees.

That rejection was correct **for the use case it was written against** (daemon restart recovery) and
is not being silently reversed here. This doc is a different use case — organizing what happens
*within* a live session, not recovering one after a crash — and it independently needs the tree shape
that use case didn't. Both are true at once: flat snapshots remain the right answer for "the daemon
died, reload what I had," and a tree-structured log is the right answer for "keep the trunk small
while the model does real work per turn." A session can have a flat snapshot taken of whatever its
current leaf's path looks like at any point — the two aren't in tension once that's seen as a
storage-shape choice, not a competing proposal. `# proposed`, supersedes nothing in that doc.

Design conversation this came out of (2026-10-15, not transcribed here) settled on: pi's **tree**
mode is the right primitive, not pi's `/fork` command. Confirmed by reading pi's actual source
(`earendil-works/pi`, `packages/coding-agent/src/core/session-manager.ts`), not assumed from its
CLI docs: every session entry has `id`/`parentId` (`:60`), appends take `parentId: this.leafId`
(e.g. `:1208`), `branchFromId` creates a new branch point (`:1615`), `getChildren(parentId)` (`:1419`)
walks the tree. It's one append-only log with a tree structure via parent pointers and a
"current leaf" cursor — not separate files per branch (that's what `/fork`, a different, user-facing
command, does — see "Relationship to manual fork" below).

## Core idea

Analogy stated directly by the user, worth keeping verbatim because it's the clearest handle on this:
**a physics test.** The test itself has succinct questions and answers — the trunk. Each question has
its own scratch pad where the actual work happens — a branch. Nobody grading the test reads the
scratch pads unless something looks wrong; they're there, attributable to the right question, not
thrown away.

```
Entry        # pkg/agentcontext or pkg/agent, exact package TBD by whoever lands this
  id string
  parentId string | null
  role "user" | "assistant" | "tool" | "system"
  content string
  toolCalls []llm.ToolCall         # present on assistant entries that called tools
  toolCallID string                 # present on tool-result entries
  createdAt time.Time

SessionLog
  entries map[string]Entry
  leafID string                     # the "current" position — appends attach here by default
  func append(e Entry) Entry        # e.parentId defaults to leafID; sets leafID = e.id
  func branchFrom(parentID string) string   # returns a NEW leaf id rooted at parentID;
                                              # does NOT move the caller's own leafID
  func getChildren(parentID string) []Entry
  func pathTo(leafID string) []Entry         # walks parentId pointers to root — this
                                               # IS "buildSessionProjection" from pi,
                                               # the live context for a model call
```

## Per-turn scratch branching

Triggered per user message, decided **retroactively**, not predicted up front — you can't know
before running a turn whether it'll need tools:

```
func handleTurn(trunk SessionLog, trunkLeaf string, userMsg string):
  branchLeaf := trunk.branchFrom(trunkLeaf)     # same tree, new leaf, trunkLeaf untouched
  "run the normal reasoning/tool-call loop exactly as it runs today, entirely against branchLeaf —
   full history up to and including userMsg is visible via pathTo(branchLeaf), nothing about the
   loop itself changes"
  finalText := "the branch's final assistant response, same as any turn produces today"
  # PostResponse hook, once the turn is actually answered:
  ? zero tool calls happened in this branch:
      "splice the plain exchange straight into trunk — append userMsg, append finalText,
       advance trunk's own leaf — discard the now-unused branchLeaf entirely"
  ? one or more tool calls happened:
      trunk.append({ role: "tool", content: "ref:" + branchLeaf, toolName: "turn_scratch" })
      trunk.append({ role: "assistant", content: finalText })
      "branchLeaf and its whole subtree stay in the log, addressable, never deleted"
```

The branch-pointer is deliberately an ordinary tool message, not a new message kind — every piece of
code that already understands tool-call/tool-result shapes (providers, truncation, the compaction
work in `compaction-v2.md`) handles it with zero new plumbing. It also means retrieval isn't a new
mechanism:

```
# a real AIFunction, registered like any other tool
func read_branch(branchID string, query string?) string
  "look up branchID in the SessionLog, return either the full subtree's content or (if query is
   given) whatever the agent decides is relevant from it — model-callable, same as any tool"
```

The model can retrieve its own earlier scratch work by calling `read_branch` whenever it actually
needs it back. No separate retrieval system; it falls out of the branch reference being a normal tool
result the model already knows how to act on.

### Per-tool-type / policy gating

"Whether branching happens is a predicate over what the branch actually did, not a fixed rule" —
e.g. a branch that only called one cheap `read_file` for a 3-line config might not be worth hiding,
while a long bash-debugging branch clearly is. Latitude on the exact predicate and its config shape;
the retroactive-decision-point above is the hard requirement, not any specific policy on top of it.

## Relationship to manual fork

Keep the vocabulary distinct — don't call this "fork." Manual, user-initiated branching (pi's
`/fork`, or whatever this SDK ends up calling the equivalent) is for a different *task*, possibly
never merged back, something the user deliberately switches into and works in for a while. This
mechanism is automatic, per-turn, invisible by default, and always resolves (merge-or-discard) before
the next trunk turn starts.

They compose on the same `SessionLog`, not in competition: manual fork is "move the active leaf
pointer to a new branch point and stay there"; turn-scratch-branching then applies exactly the same
way to whichever leaf is currently active — a manually-forked branch gets its own per-turn scratch
branches off of it, recursively, exactly like the trunk does. **This is not a trunk-only feature** —
anywhere in the tree that's acting as someone's "current spine" gets the same treatment. Concretely,
this means compaction (`compaction-v2.md`) has to be generic over "any sufficiently turn-heavy spine,"
not special-cased to one designated root path — every manually-forked branch that lives long enough
needs its own independent compaction eventually, on its own timeline.

## Tool-call concurrency (pi's other real gap vs. this SDK)

Unrelated to branching mechanically, grouped here because it's the other `AgentBase`-loop-shape
change from the same survey. `executeAgentLoop`'s tool loop (`base.go:392-524`) executes every tool
call in a batch strictly sequentially today. Direct port of pi's model
(`packages/agent` README:111-120, read in full during the survey): "parallel" execution is the
default — every call in a batch runs concurrently, `EventToolResult` fires in whatever order calls
actually finish, but results are written into history in the **assistant's original request order**,
so the model never sees results out of order even though they completed out of order. "sequential" is
opt-in, settable globally or per-tool; a batch containing any sequential-tagged tool runs the *whole*
batch sequentially. No correctness reason to default to serial for bash/read/web-heavy workloads; pure
wall-clock cost today.

## Honest accounting: KV-cache cost

Said plainly because it matters and is easy to oversell: **this is not cache-free.** Prefix-caching
inference servers (vLLM's automatic prefix caching, llama.cpp, etc.) match tokens from the start of a
sequence and hit-or-miss at the first point of divergence — no partial credit for "the change was
small." The branch's own last forward pass ran over `[trunk ... + real tool calls + real results +
finalText]`. The trunk's merge writes `[trunk ... + ref-message + finalText]` — a sequence that was
**never itself the subject of any forward pass**; it's assembled client-side from pieces. So every
trunk turn's merge is a guaranteed cache-miss on that (small, bounded) block, forever — this does not
get cheaper over time, it's a fixed, recurring cost per turn.

What *is* true, and is the actual reason this is still the right trade: compaction
(`compaction-v2.md`) **retroactively rewrites already-committed history**, busting a cache that was
valid up to that moment, for a large, unpredictably-timed chunk, whenever it fires. This mechanism
never touches already-committed trunk content — once a merge lands it's stable forever. So the
comparison is: one large, unpredictable cache-bust (status quo, compaction) vs. one small, guaranteed,
*predictable* cache-miss every turn (this). The latter is the better trade, but it is a trade, not a
free lunch — don't describe this to anyone as "doesn't affect KV cache."

If the serving backend supports explicit cache breakpoints (rather than automatic prefix hashing),
this is manageable further: place a new breakpoint right after each merge, so only the small new block
is ever paid for at full-compute rates and everything before it stays a cache read.

## Open questions

- `??` Whether pi's tree is scoped per-session-file or shared globally across `/fork`s was not
  confirmed during the survey — check `session-manager.ts` more closely before this becomes real
  implementation; it affects whether `SessionLog` is one-per-top-level-session or something broader.
- `??` Addressing scheme for `read_branch` at depth: "the branch for query 2" is ambiguous once
  there's more than one level of manual fork in play (query 2 of the trunk? of the current fork?).
  Needs real IDs/labels scoped unambiguously — not resolved here, flagged because it'll bite the UX of
  retrieval specifically, not the storage model.
- `??` Pruning/archival policy for a long-lived, heavily-forked tree. Nothing here ever deletes a
  branch; that's deliberate (lossless, same spirit as the AnythingLLM retrieval note in the survey),
  but "nobody will ever retrieve this one" needs an eventual answer that isn't designed here.

## Relationship to other design docs

- `session-dispatch.md` — see "Context" above; additive, not a replacement. Its `Session`/steering
  queue design is unaffected; a steered message mid-branch-execution behaves exactly as that doc
  describes, just scoped to whichever leaf is currently active.
- `compaction-v2.md` — compaction must be spine-generic, as stated above; both docs were written with
  that constraint in mind, but it hasn't been verified against real code yet since neither is
  implemented. Don't assume it from this prose.
