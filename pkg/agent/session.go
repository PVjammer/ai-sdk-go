package agent

import (
	"context"
	"sync"

	"github.com/pvjammer/ai-sdk-go/pkg/llm"
)

// sessionSeeder is implemented by AgentBase (and so by any concrete agent embedding it).
// Turn-branching needs it to prime and read back a scratch session; without it, a Session with
// a SessionLog attached still works, it just never branches — see sendBranched.
type sessionSeeder interface {
	SeedSession(sessionID string, history []llm.ChatMessage)
	SessionHistory(sessionID string) []llm.ChatMessage
	ClearSession(sessionID string)
}

// Session wraps an AIAgent and owns one session's lifecycle: steering, follow-up queuing, and
// (optionally) turn-scratch branching against a SessionLog.
//
// See docs/design/session-dispatch.md for the Session/steering/follow-up design this
// implements, and docs/design/turn-branching.md for SessionLog and the branching behavior.
//
// A Session takes ownership of the wrapped agent's hooks for its own lifetime (it layers a
// steering PreModel hook over whatever was already installed, once, at construction). Don't
// share one AIAgent instance across multiple Sessions — AgentBase's hooks are a single field
// per agent instance, not per session, so two Sessions sharing an agent would stomp on each
// other's steering. Give each Session its own agent instance.
type Session struct {
	id     string
	agent  AIAgent
	seeder sessionSeeder // nil if agent doesn't implement it — branching degrades gracefully

	log  *SessionLog // nil disables branching entirely
	leaf string      // this spine's current position in log; meaningless if log == nil

	steerCh chan string

	followUpMu sync.Mutex
	followUp   []string
}

// SessionConfig configures a new Session.
type SessionConfig struct {
	// Log, if non-nil, enables turn-scratch branching: each Send retroactively decides whether
	// that turn's work stays directly on this spine (no tool calls happened) or gets a branch
	// with a tool-message reference merged back (tool calls happened) — see
	// docs/design/turn-branching.md. Leaf is this spine's starting position ("" for a fresh log,
	// or an existing entry ID to start as a manually-forked spine off of it).
	Log  *SessionLog
	Leaf string

	// SteerBuffer sizes the steering channel (default 4). Steer is always non-blocking; once
	// full, the newest Steer call is dropped rather than blocking the caller.
	SteerBuffer int
}

// NewSession creates a session wrapping a. id is used as the sessionID passed to a.Run when
// branching is disabled or unsupported, and as a prefix for scratch session IDs when turn
// branching runs.
func NewSession(id string, a AIAgent, cfg SessionConfig) *Session {
	if cfg.SteerBuffer <= 0 {
		cfg.SteerBuffer = 4
	}
	s := &Session{
		id:      id,
		agent:   a,
		log:     cfg.Log,
		leaf:    cfg.Leaf,
		steerCh: make(chan string, cfg.SteerBuffer),
	}
	s.seeder, _ = a.(sessionSeeder)

	// Install steering once, layered over whatever PreModel the agent already had (compaction,
	// TokenLimit, goal-pinning, ...) — steering runs FIRST so a steer message is already in the
	// list before anything else decides what to trim or summarize.
	base := a.GetHooks()
	a.SetHooks(chainPreModel(s.steeringHook(), base))

	return s
}

// ID returns the session identifier.
func (s *Session) ID() string { return s.id }

// Leaf returns this spine's current position in its SessionLog. Meaningless if the session has
// no SessionLog attached.
func (s *Session) Leaf() string { return s.leaf }

// Log returns the attached SessionLog, or nil if branching is disabled.
func (s *Session) Log() *SessionLog { return s.log }

// Steer injects a message to be spliced in as soon as the current tool-call batch finishes,
// redirecting the agent without aborting the current turn. Non-blocking: if the steer buffer is
// full, the newest call is dropped rather than blocking the caller. If no turn is currently
// running when the message is consumed, it's simply prepended to the next turn's first LLM call
// like an extra user message — harmless, just not "mid-turn" in that case.
func (s *Session) Steer(message string) {
	select {
	case s.steerCh <- message:
	default:
		// Buffer full — drop rather than block. A caller steering faster than the agent can
		// consume steers has bigger problems than one dropped redirect.
	}
}

// SendFollowUp queues message to be delivered as a new turn once the current Send's event
// channel would otherwise close.
func (s *Session) SendFollowUp(message string) {
	s.followUpMu.Lock()
	defer s.followUpMu.Unlock()
	s.followUp = append(s.followUp, message)
}

func (s *Session) popFollowUp() (string, bool) {
	s.followUpMu.Lock()
	defer s.followUpMu.Unlock()
	if len(s.followUp) == 0 {
		return "", false
	}
	next := s.followUp[0]
	s.followUp = s.followUp[1:]
	return next, true
}

// Send delivers message and returns an event channel that closes once the turn — and any
// follow-ups queued via SendFollowUp, delivered as subsequent turns on the same channel — have
// completed.
func (s *Session) Send(ctx context.Context, message string) <-chan Event {
	out := make(chan Event, 50)
	go func() {
		defer close(out)
		s.sendOnce(ctx, message, out)
		for {
			next, ok := s.popFollowUp()
			if !ok {
				return
			}
			s.sendOnce(ctx, next, out)
		}
	}()
	return out
}

// steeringHook returns a PreModel hook that non-blockingly drains s.steerCh and appends each
// pending steer message as a user message before the next LLM call. PreModel fires before every
// LLM call in AgentBase's loop, including between tool-call rounds, so this gives exactly
// "between tool calls, before the next LLM call" without any change to the loop itself.
func (s *Session) steeringHook() AgentHooks {
	return AgentHooks{
		PreModel: func(ctx context.Context, messages []llm.ChatMessage) ([]llm.ChatMessage, error) {
			for {
				select {
				case m := <-s.steerCh:
					messages = append(messages, llm.ChatMessage{Role: "user", Content: m})
				default:
					return messages, nil
				}
			}
		},
	}
}

// chainPreModel returns hooks identical to base except PreModel runs first's transform, then
// base's own PreModel (if any), in sequence. A two-hook special case rather than a general
// combinator — pkg/agent/hooks.Combine can't be imported here (it imports this package).
func chainPreModel(first, base AgentHooks) AgentHooks {
	combined := base
	combined.PreModel = func(ctx context.Context, messages []llm.ChatMessage) ([]llm.ChatMessage, error) {
		var err error
		if first.PreModel != nil {
			messages, err = first.PreModel(ctx, messages)
			if err != nil {
				return messages, err
			}
		}
		if base.PreModel != nil {
			return base.PreModel(ctx, messages)
		}
		return messages, nil
	}
	return combined
}

// sendOnce runs exactly one turn (no follow-up draining) and forwards its events to out.
func (s *Session) sendOnce(ctx context.Context, message string, out chan<- Event) {
	if s.log == nil || s.seeder == nil {
		forward(s.agent.Run(ctx, s.id, message), out)
		return
	}
	s.sendBranched(ctx, message, out)
}

func forward(events <-chan Event, out chan<- Event) {
	for ev := range events {
		out <- ev
	}
}

// sendBranched implements turn-scratch branching (docs/design/turn-branching.md): the turn runs
// against a scratch session seeded from this spine's current path, and the decision to splice
// (no tool calls happened — the plain exchange joins the spine directly) or merge (tool calls
// happened — the spine gets a tool-message branch reference plus the real final answer, and the
// branch itself stays in the log, addressable, never deleted) is made retroactively, after the
// turn has actually run, since there's no reliable way to know up front whether a turn will need
// tools.
func (s *Session) sendBranched(ctx context.Context, message string, out chan<- Event) {
	trunkLeaf := s.leaf
	scratchID := s.id + "#scratch-" + newEntryID()
	seedHistory := EntriesToMessages(s.log.Path(trunkLeaf))
	s.seeder.SeedSession(scratchID, seedHistory)

	toolCalls := 0
	var finalText string
	for ev := range s.agent.Run(ctx, scratchID, message) {
		switch ev.Type {
		case EventToolCall:
			toolCalls++
		case EventResponse:
			if r, ok := ev.Data.(ResponseEvent); ok && r.Done {
				finalText = r.Content
			}
		}
		out <- ev
	}

	full := s.seeder.SessionHistory(scratchID)
	s.seeder.ClearSession(scratchID)

	// full = seedHistory + this turn's new messages (the user message, any assistant/tool
	// rounds, the final assistant message). Replay only the new portion into the log as real
	// Entries — the branch is always recorded, regardless of which outcome below applies.
	newMsgs := full
	if len(seedHistory) <= len(full) {
		newMsgs = full[len(seedHistory):]
	}

	cursor := trunkLeaf
	var branchRoot string
	for i, m := range newMsgs {
		e := s.log.Append(cursor, MessageToEntry(m))
		if i == 0 {
			branchRoot = e.ID
		}
		cursor = e.ID
	}

	if toolCalls == 0 {
		// No scratch work worth hiding — the exchange joins the spine directly.
		s.leaf = cursor
		return
	}

	// Merge: the spine gets a tool-message branch reference, not the branch's own content. The
	// branch (everything just appended above, rooted at trunkLeaf) stays in the log as a
	// sibling subtree, retrievable later via a read_branch(branchRoot) tool call.
	ref := s.log.Append(trunkLeaf, Entry{
		Role:     "tool",
		ToolName: "turn_scratch",
		Content:  "ref:" + branchRoot,
	})
	ans := s.log.Append(ref.ID, Entry{Role: "assistant", Content: finalText})
	s.leaf = ans.ID
}

// ToolCallRecord is one tool invocation captured by AccumulateEvents.
type ToolCallRecord struct {
	ToolName string
	Input    interface{}
	Output   interface{}
	Error    error
}

// RunResult is the accumulated outcome of draining an event channel to completion.
type RunResult struct {
	Response  string
	ToolCalls []ToolCallRecord
	Error     error
	Truncated bool // true if the agent hit ErrMaxIterations
}

// AccumulateEvents drains an event channel and returns a RunResult. Blocks until the channel
// closes. Callers that need live streaming should consume the channel directly instead.
//
// Tool-call Input/Output pairing is best-effort: EventToolResult carries no call ID to
// correlate against, so pairing is FIFO per tool name. Under Phase 3's concurrent tool
// execution, two in-flight calls to the SAME tool name in one batch can have their
// Input/Output crossed. Exact per-call correlation would need an ID on ToolResultEvent, which
// doesn't exist yet — flagged, not fixed here.
func AccumulateEvents(ch <-chan Event) RunResult {
	var r RunResult
	pendingByTool := make(map[string][]int) // toolName -> indices into r.ToolCalls awaiting a result

	for ev := range ch {
		switch ev.Type {
		case EventToolCall:
			tc := ev.Data.(ToolCallEvent)
			idx := len(r.ToolCalls)
			r.ToolCalls = append(r.ToolCalls, ToolCallRecord{ToolName: tc.ToolName, Input: tc.Input})
			pendingByTool[tc.ToolName] = append(pendingByTool[tc.ToolName], idx)

		case EventToolResult:
			tr := ev.Data.(ToolResultEvent)
			if idxs := pendingByTool[tr.ToolName]; len(idxs) > 0 {
				idx := idxs[0]
				pendingByTool[tr.ToolName] = idxs[1:]
				r.ToolCalls[idx].Output = tr.Output
				r.ToolCalls[idx].Error = tr.Error
			}

		case EventResponse:
			re := ev.Data.(ResponseEvent)
			if re.Done {
				r.Response = re.Content
			}

		case EventError:
			ee := ev.Data.(ErrorEvent)
			r.Error = ee.Error
			if ee.Error == ErrMaxIterations {
				r.Truncated = true
			}
		}
	}
	return r
}
