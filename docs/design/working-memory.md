# Working Memory — Design Idea

## Problem

The agent loop (`AgentBase.BaseRun`) appends every tool result verbatim to
`session.history`. For tools that return large content (file reads, API
responses, database query results), this causes context to grow unboundedly
across iterations. The LLM ends up carrying raw content it already acted on,
consuming context window for no benefit.

Compression via `PostTool` hook is not a solution: the hook fires before the
result enters history, so the LLM never sees the raw content at all — defeating
the purpose of the tool call.

## Idea: Working Memory as a Distinct Context Type

Introduce a `WorkingMemory` store that sits alongside `session.history` but is
managed separately.

### How it works

1. **Tool result → working memory**: instead of appending raw content to
   history, the result is stored in a keyed working memory store
   (`wm:<uuid>`). History receives a compact reference:
   ```json
   {"ref": "wm:abc123", "preview": "first 3 lines...", "total_lines": 47}
   ```

2. **PreModel hook injects working memory**: before each LLM call, scan
   history for `wm:` refs and inject the referenced content into the current
   message list. The LLM sees full content when reasoning; history stays
   compact.

3. **Flush signal**: working memory items are dropped when the agent is done
   with them. Three possible mechanisms:
   - **Explicit**: agent calls a `wm_release(ref)` tool
   - **TTL**: auto-flush after N turns of non-use
   - **Phase boundary**: caller flushes all working memory at a known
     transition point (e.g. between survey and write phases)

### Result

- Agent has full fidelity when it needs content
- History only retains compact refs — stays readable and flat
- Working memory is explicitly scoped and bounded
- Large content can be backed by disk for effectively unlimited capacity

## Relation to Existing SDK Primitives

| Existing | Role |
|---|---|
| `session.history` | Persistent conversation record (compact refs only) |
| `agentcontext.Extended` | Per-session working memory with auto-compression |
| `PostTool` hook | Still useful for transforming results before working memory storage |
| `PreModel` hook | Injection point for working memory into current message list |

`agentcontext.Extended` already exists for per-session working context with
compression hooks. Working memory could be implemented as a specialization of
it — or as a lighter-weight keyed store if `Extended` is too heavy.

## Prior Art

- **LangChain DeepAgents** `SummarizationToolMiddleware`: offloads tool results
  >20k tokens to a virtual filesystem, replaces history entry with path +
  10-line preview. Model can re-read via `read_file` tool. Closest existing
  implementation of this pattern.
- **Anthropic context editing** (`clear_tool_uses_20250919`): server-side
  clearing of old tool results, replaced with placeholder. Client retains full
  history locally.
- **Anthropic "Effective context engineering"**: recommends tools return only
  high-signal fields; store identifiers, not content.

## Candidate Use Cases in This Repo

- `curate` workflow: file reads during survey phase; flush before write phase
- `compile` workflow: reading multiple wiki categories before synthesis
- `rebuild` workflow: reading all files in a wiki before reorganization
- Any future workflow that reads external content (APIs, databases) before acting

## Open Questions

- Should working memory be part of `AgentBase` or a separate injectable
  component?
- Should the `PreModel` hook receive working memory as a parameter, or should
  working memory injection be a dedicated hook slot?
- Is the explicit `wm_release` tool ergonomic, or does phase-boundary flushing
  cover most real cases?
- Should working memory back large items to disk automatically (like LangChain's
  virtual filesystem)?

## Resolution (2026-10-15), see `compaction-v2.md`

Phase-boundary flushing, with the boundary being the turn itself: `PostResponse` is the flush point.
If nothing later in the same turn needed a `wm:` ref re-injected in full, the ref stands permanently
instead of being re-expanded on the next turn — no explicit `wm_release` needed as the *default*
path (still fine as an escape hatch for an agent that knows it's done with something mid-turn).
This was motivated by a separate, later ask ("replace raw tool results with a compact version once
the turn's answered") that turned out to be this same open question, just answered from the other
direction. Doesn't resolve the disk-backing question or the `AgentBase`-vs-separate-component question
above — those are still open.
