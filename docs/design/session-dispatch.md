# Session Object & Dispatch-Style Agent Improvements

## Context

The current `AgentBase.BaseRun()` API is fire-and-forget: the caller passes a
sessionID string and receives a channel of events. This works for single-turn
workflows but becomes awkward for dispatch-style orchestrators that manage
multiple concurrent sessions over time.

Inspired by pi coding agent's `createAgentSession()` SDK
(github.com/badlogic/pi-mono/tree/main/packages/coding-agent).

---

## 1. Session as a First-Class Object

### Problem

`BaseRun` is stateless from the caller's perspective. To maintain multiple
concurrent sessions, the dispatcher must track sessionIDs externally and
re-pass them on every call. There is no object representing a session's
lifecycle, no way to close one explicitly, and no handle for sending
subsequent messages to the same session.

### Design

Introduce a `Session` type that wraps `AgentBase` and owns a single
session's lifecycle:

```go
// pkg/agent/session.go

type Session struct {
    id       string
    agent    *AgentBase
    ctx      context.Context
    cancel   context.CancelFunc
    steerCh  chan steerMsg     // mid-execution steering (see §2)
}

type SessionConfig struct {
    SystemPrompt  string
    MaxIterations int
    Verbose       bool
    Hooks         AgentHooks
}

// NewSession creates a session. The session is idle until Send is called.
func NewSession(ctx context.Context, id string, agent *AgentBase, cfg SessionConfig) *Session

// Send delivers a message and returns an event channel that closes when
// the agent reaches a final response or error.
func (s *Session) Send(message string) <-chan Event

// Steer injects a message between tool calls during an active Send.
// If no Send is active, the message is queued as a follow-up.
// See §2 for full steering semantics.
func (s *Session) Steer(message string)

// Close releases session history and cancels any active run.
func (s *Session) Close()

// ID returns the session identifier.
func (s *Session) ID() string
```

### Dispatcher usage

```go
// Dispatcher holds sessions by ID
sessions := map[string]*agent.Session{}

// On new connection
sess := agent.NewSession(ctx, clientID, agentBase, cfg)
sessions[clientID] = sess

// On message received
events := sess.Send(userMessage)
go streamToClient(clientID, events)

// On disconnect
sessions[clientID].Close()
delete(sessions, clientID)
```

---

## 2. Steering and Follow-Up Message Queues

### Problem

Once `BaseRun` is called, there is no way to inject new information until it
completes. For dispatch-style workflows, two distinct interrupt patterns arise:

- **Redirect**: user sends a correction while the agent is mid-execution
  ("actually focus on X, not Y"). Should take effect at the next tool call
  boundary, not after the current run finishes.
- **Follow-up**: user sends a new request that should be queued until the
  agent finishes its current task, then delivered as a new turn.

### Design

Two queues on the `Session` object, matching pi's model:

**Steering messages** — delivered between tool calls during an active run.
The agent loop checks the steer channel after each tool call batch. If a
steer message is present, it is injected into the message list before the
next LLM call. This redirects the agent without aborting the current run.

```go
// Inject mid-execution (non-blocking; queued if channel is full)
sess.Steer("focus on the auth module, skip the tests for now")
```

**Follow-up messages** — queued while a run is active, delivered
automatically when the event channel closes. The dispatcher does not need
to detect completion and re-call Send manually.

```go
// Queued if a run is active; delivered on next idle turn
sess.SendFollowUp("now run the tests")
```

### Agent loop changes

In `executeAgentLoop`, after executing each tool call batch and before the
next LLM call:

```go
// Check for steering message (non-blocking)
select {
case steer := <-s.steerCh:
    session.history = append(session.history, llm.ChatMessage{
        Role:    "user",
        Content: steer.message,
    })
default:
}
```

On run completion, drain the follow-up queue and start the next turn if
non-empty.

---

## 3. `AccumulateEvents` / `RunResult`

### Problem

Every caller of `BaseRun` (curate, compile, rebuild, chat) writes the same
event accumulation loop:

```go
var summary string
for event := range events {
    switch event.Type {
    case agent.EventResponse:
        summary += event.Data.(agent.ResponseEvent).Content
    case agent.EventError:
        return event.Data.(agent.ErrorEvent).Error
    }
}
```

This is boilerplate with no value differentiation between callers.

### Design

```go
// pkg/agent/result.go

type ToolCallRecord struct {
    ToolName string
    Input    interface{}
    Output   interface{}
    Error    error
}

type RunResult struct {
    Response  string           // final text response
    ToolCalls []ToolCallRecord // all tool calls made, in order
    Error     error            // first error encountered, if any
    Truncated bool             // true if maxIterations was reached
}

// AccumulateEvents drains an event channel and returns a RunResult.
// Blocks until the channel closes.
func AccumulateEvents(ch <-chan Event) RunResult
```

Callers that need streaming still consume the channel directly. Callers
that only need the final result use `AccumulateEvents`.

---

## 4. Session Snapshot / Restore

### Problem

The daemon process may restart (crash, deploy, OS reboot). All in-memory
session history is lost. Reconnecting clients start fresh with no context.

### Design

Simple JSON snapshot: serialize `session.history` to disk on `Close()`,
restore on `NewSession()` if a snapshot exists.

```go
type SessionSnapshot struct {
    ID        string            `json:"id"`
    History   []llm.ChatMessage `json:"history"`
    CreatedAt time.Time         `json:"created_at"`
    UpdatedAt time.Time         `json:"updated_at"`
}

// SnapshotStore persists and loads session snapshots.
type SnapshotStore interface {
    Save(snap *SessionSnapshot) error
    Load(id string) (*SessionSnapshot, error)
    Delete(id string) error
    List() ([]string, error)
}

// FilesystemSnapshotStore is the default implementation.
// Stores one JSON file per session at <dir>/<id>.json.
func NewFilesystemSnapshotStore(dir string) SnapshotStore
```

`NewSession` accepts an optional `SnapshotStore`. If provided:
- On creation: load existing snapshot if present
- On `Close()`: save snapshot
- Sessions older than a configurable TTL are pruned on load

This intentionally does not implement pi's JSONL tree branching. Flat
snapshots cover the daemon restart use case without the complexity of
navigable history trees.

---

## Implementation Order

1. `AccumulateEvents` / `RunResult` — low effort, high immediate value;
   removes boilerplate in personal-assistant today
2. `Session` object — medium effort; unblocks the dispatch pattern
3. Steering queue — medium effort; requires agent loop changes
4. Session snapshots — low effort once `Session` object exists; add
   `SnapshotStore` as an optional field

---

## Relationship to Other Design Docs

- `working-memory.md` — the `Session` object is the right place to attach
  a `WorkingMemory` store; the `PreModel` hook on the session would inject
  working memory refs into each turn's message list
- `PA_WIKI_IMPROVEMENTS.md §8.3-8.5` — `AccumulateEvents` covers §8.4;
  `Session` object supersedes the multi-session coordinator (§8.3) by
  making single-session management cleaner first
