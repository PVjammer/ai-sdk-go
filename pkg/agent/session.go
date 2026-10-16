package agent

import (
	"context"
	"fmt"
	"regexp"
	"sort"
	"strings"
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
	SessionUsage(sessionID string) (promptTokens, completionTokens int)
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
//
// Send is not designed for concurrent turns on the same Session: two overlapping Send calls
// would both read the same starting leaf and race to append/advance it when branching is
// enabled. Steer and SendFollowUp are safe to call while a Send is in flight — that's their
// purpose — but don't call Send again until the previous one's channel has closed.
type Session struct {
	id     string
	agent  AIAgent
	seeder sessionSeeder // nil if agent doesn't implement it — branching degrades gracefully

	log *SessionLog // nil disables branching entirely

	leafMu sync.Mutex
	leaf   string // this spine's current position in log; meaningless if log == nil

	steerCh chan string

	followUpMu sync.Mutex
	followUp   []string

	usageMu    sync.Mutex
	lastUsage  TurnUsage // the most recently completed Send's own token cost, isolated from
	                     // every other turn before or after it — see sendOnce/sendBranched.
}

// TurnUsage is one turn's own prompt/completion token cost, summed across every LLM round that
// turn made (tool-call rounds included, not just the final response).
type TurnUsage struct {
	PromptTokens     int
	CompletionTokens int
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
// no SessionLog attached. Safe to call while a Send is in flight — reads the same
// mutex-guarded field sendBranched writes.
func (s *Session) Leaf() string {
	s.leafMu.Lock()
	defer s.leafMu.Unlock()
	return s.leaf
}

// Log returns the attached SessionLog, or nil if branching is disabled.
func (s *Session) Log() *SessionLog { return s.log }

// LastTurnUsage returns the token cost of the most recently completed Send — not a running
// total across the whole session, just that one turn, isolated the same way regardless of
// whether branching is on (where the scratch session already starts at zero) or off (where the
// main session's running total is snapshotted before and after, see sendOnce).
func (s *Session) LastTurnUsage() TurnUsage {
	s.usageMu.Lock()
	defer s.usageMu.Unlock()
	return s.lastUsage
}

func (s *Session) setLastUsage(u TurnUsage) {
	s.usageMu.Lock()
	defer s.usageMu.Unlock()
	s.lastUsage = u
}

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
		// Flat mode's session lives for the whole conversation, not just one turn, so
		// isolating this turn's own cost means snapshotting the running total before and
		// after — unlike sendBranched's scratch session, which already starts at zero.
		var before TurnUsage
		if s.seeder != nil { // nil only if the agent genuinely doesn't implement SessionUsage
			before.PromptTokens, before.CompletionTokens = s.seeder.SessionUsage(s.id)
		}
		forward(s.agent.Run(ctx, s.id, message), out)
		if s.seeder != nil {
			afterP, afterC := s.seeder.SessionUsage(s.id)
			s.setLastUsage(TurnUsage{PromptTokens: afterP - before.PromptTokens, CompletionTokens: afterC - before.CompletionTokens})
		}
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
	trunkLeaf := s.Leaf()
	scratchID := s.id + "#scratch-" + newEntryID()
	seedHistory := EntriesToMessages(s.log.Path(trunkLeaf))
	s.seeder.SeedSession(scratchID, seedHistory)

	toolCalls := 0
	var calls []ToolCallEvent
	var finalText string
	for ev := range s.agent.Run(ctx, scratchID, message) {
		switch ev.Type {
		case EventToolCall:
			toolCalls++
			if tc, ok := ev.Data.(ToolCallEvent); ok {
				calls = append(calls, tc)
			}
		case EventResponse:
			if r, ok := ev.Data.(ResponseEvent); ok && r.Done {
				finalText = r.Content
			}
		}
		out <- ev
	}

	full := s.seeder.SessionHistory(scratchID)
	// The scratch session started at zero (SeedSession always creates a fresh *agentSession
	// for a never-before-seen scratchID), so its running total right now already IS this
	// turn's own cost — no before/after delta needed, unlike sendOnce's flat-session case.
	usageP, usageC := s.seeder.SessionUsage(scratchID)
	s.setLastUsage(TurnUsage{PromptTokens: usageP, CompletionTokens: usageC})
	s.seeder.ClearSession(scratchID)

	// full = seedHistory + this turn's new messages (the user message, any assistant/tool
	// rounds, the final assistant message). Replay only the new portion into the log as real
	// Entries — the branch is always recorded, regardless of which outcome below applies.
	//
	// This is a length-based diff. It used to be unsafe against anything that rewrites the
	// scratch session's history mid-turn: SeedSession (called by Compaction's PreModel from
	// inside executeAgentLoop) used to allocate a brand-new *agentSession and swap the map
	// entry, silently detaching the *agentSession pointer executeAgentLoop already held for
	// the rest of the turn — every tool call and the final answer appended afterward went to
	// an object SessionHistory could no longer see. SeedSession now mutates an existing
	// session's history in place instead (base.go), which fixes that at the root: full is
	// always complete now. What's left here is a narrower, honest case: compaction can
	// legitimately shrink the message COUNT below len(seedHistory) by replacing many old
	// messages with one summary, even though real new content exists after that point. In
	// that case full[len(seedHistory):] isn't the right slice either, so fall back to
	// replaying the whole of full — it may duplicate a little content the trunk already has,
	// which is harmless, rather than trying to be clever about exactly where the new part
	// starts.
	newMsgs := full
	if len(seedHistory) <= len(full) {
		newMsgs = full[len(seedHistory):]
		if len(newMsgs) == 0 && toolCalls > 0 {
			newMsgs = full
		}
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
		s.setLeaf(cursor)
		return
	}

	// Merge: the spine gets a tool-message branch reference, not the branch's own content. The
	// branch (everything just appended above, rooted at trunkLeaf) stays in the log as a
	// sibling subtree, retrievable later via read_branch.
	//
	// Round 1 of this made the reference self-describing but still keyed on the raw branch ID
	// (a 16-char random hex string) as the thing the model had to notice and copy — which turned
	// out to be its own failure mode (a model that DID try to retrieve copied a literal example
	// ID from a prompt instead of the real one). This round gives the model a short, sequential
	// label (B1, B2, ...) to use instead — easier to get right than 16 hex characters — and
	// pushes full discovery onto a separate list_branches tool/catalog rather than relying
	// purely on the model noticing and re-parsing this inline text, which degrades once history
	// is long or compacted. The real ID still lives in the content, in a machine-parseable
	// {ref:<id>} marker (ParseBranchRef), for read_branch to resolve a label against — the model
	// is never expected to read or use that part directly.
	label := fmt.Sprintf("B%d", countBranches(s.log)+1)
	did, mentions := summarizeToolCalls(calls)
	content := fmt.Sprintf("branch %s: did %s", label, did)
	if mentions != "" {
		content += " [mentions: " + mentions + "]"
	}
	content += " {ref:" + branchRoot + "}"

	ref := s.log.Append(trunkLeaf, Entry{
		Role:     "tool",
		ToolName: "turn_scratch",
		Content:  content,
	})
	ans := s.log.Append(ref.ID, Entry{Role: "assistant", Content: finalText})
	s.setLeaf(ans.ID)
}

// setLeaf updates this spine's current log position under leafMu. See the field doc comment on
// Session.leaf — reads go through Leaf(), which uses the same lock.
func (s *Session) setLeaf(id string) {
	s.leafMu.Lock()
	defer s.leafMu.Unlock()
	s.leaf = id
}

// summarizeToolCalls renders a branch's tool calls as (did, mentions): did is a short
// "tool(firstArg)" list for a quick sense of WHAT HAPPENED (capped at 6 calls so one very chatty
// turn can't produce an unbounded trunk entry); mentions is every distinct string-ish argument
// value across ALL calls and args, deduplicated — so matching "what was beta's value" against
// a breadcrumb is closer to a plain string match than an inference the model has to make from
// tool names alone.
func summarizeToolCalls(calls []ToolCallEvent) (did, mentions string) {
	if len(calls) == 0 {
		return "nothing — plain exchange", ""
	}
	const maxShown = 6
	shown := calls
	truncated := false
	if len(shown) > maxShown {
		shown = shown[:maxShown]
		truncated = true
	}
	parts := make([]string, 0, len(shown))
	seen := make(map[string]bool)
	var mentionList []string
	for _, c := range calls { // mentions scans ALL calls, not just the shown/capped subset
		for _, v := range stringArgValues(c.Input) {
			if !seen[v] {
				seen[v] = true
				mentionList = append(mentionList, v)
			}
		}
	}
	for _, c := range shown {
		parts = append(parts, c.ToolName+"("+firstArgSummary(c.Input)+")")
	}
	did = strings.Join(parts, ", ")
	if truncated {
		did += ", ..."
	}
	return did, strings.Join(mentionList, ", ")
}

// firstArgSummary returns "key=value" for the first argument (by sorted key) of a tool call's
// input map, or "" if Input isn't a map or is empty — just enough to distinguish
// "lookup_fact(topic=alpha)" from "lookup_fact(topic=beta)", not a full argument dump.
func firstArgSummary(input interface{}) string {
	m, ok := input.(map[string]interface{})
	if !ok || len(m) == 0 {
		return ""
	}
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	k := keys[0]
	return k + "=" + truncateVal(fmt.Sprintf("%v", m[k]))
}

// freeTextArgKeys are argument names conventionally holding prose/payload content rather than a
// short identifier — excluded from the mentions list, which is meant to be scannable labels
// (topics, paths, names), not truncated paragraph fragments.
var freeTextArgKeys = map[string]bool{"content": true, "text": true, "body": true, "message": true}

// stringArgValues returns every short, identifier-shaped string argument value in a tool call's
// input map, in sorted-key order, for summarizeToolCalls' mentions list — skipping
// freeTextArgKeys and anything too long to plausibly be an identifier rather than prose.
func stringArgValues(input interface{}) []string {
	m, ok := input.(map[string]interface{})
	if !ok {
		return nil
	}
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	out := make([]string, 0, len(keys))
	for _, k := range keys {
		if freeTextArgKeys[strings.ToLower(k)] {
			continue
		}
		if v, ok := m[k].(string); ok && v != "" && len(v) <= 60 {
			out = append(out, v)
		}
	}
	return out
}

func truncateVal(v string) string {
	if len(v) > 40 {
		return v[:40] + "..."
	}
	return v
}

// branchRefPattern extracts the real branch ID from a turn_scratch entry's trailing {ref:<id>}
// marker — the only machine-parseable part of the content; everything else is model-facing
// prose the model is never expected to parse structurally.
var branchRefPattern = regexp.MustCompile(`\{ref:([^}]+)\}`)

// ParseBranchRef extracts the real branch (root entry) ID from a turn_scratch reference's
// Content. ok is false if content isn't a recognizable ref.
func ParseBranchRef(content string) (id string, ok bool) {
	m := branchRefPattern.FindStringSubmatch(content)
	if m == nil {
		return "", false
	}
	return m[1], true
}

// branchLabelPattern extracts the short "B<n>" label a model is told to use with read_branch,
// from a turn_scratch entry's leading "branch B<n>:" text.
var branchLabelPattern = regexp.MustCompile(`^branch (B\d+):`)

// ParseBranchLabel extracts a turn_scratch reference's short label (e.g. "B3"). ok is false if
// content doesn't start with the expected "branch B<n>:" prefix.
func ParseBranchLabel(content string) (label string, ok bool) {
	m := branchLabelPattern.FindStringSubmatch(content)
	if m == nil {
		return "", false
	}
	return m[1], true
}

// BranchInfo is one entry in ListBranches' catalog: a label a model can pass to read_branch,
// the real underlying ID, and the reference's full display text (breadcrumb + mentions).
type BranchInfo struct {
	Label   string
	RootID  string
	Content string
}

// countBranches returns how many turn_scratch references already exist in log — used to assign
// the next one's label deterministically (see sendBranched). Recomputed from the log itself
// rather than an in-memory counter, so labels stay consistent across process restarts (a fresh
// process has no memory of a prior counter, but the persisted log still has every past ref).
func countBranches(log *SessionLog) int {
	n := 0
	for _, e := range log.Entries() {
		if e.Role == "tool" && e.ToolName == "turn_scratch" {
			n++
		}
	}
	return n
}

// ListBranches catalogs every branch reference in log, in creation order, labeled B1, B2, ... —
// a model's primary way to discover what's retrievable without having to notice and re-parse an
// inline breadcrumb from potentially-distant (or compacted-away) history. Register this as a
// tool alongside read_branch; see examples/tui-agent for the pairing.
func ListBranches(log *SessionLog) []BranchInfo {
	var refs []Entry
	for _, e := range log.Entries() {
		if e.Role == "tool" && e.ToolName == "turn_scratch" {
			refs = append(refs, e)
		}
	}
	sort.Slice(refs, func(i, j int) bool { return refs[i].CreatedAt.Before(refs[j].CreatedAt) })

	out := make([]BranchInfo, 0, len(refs))
	for i, e := range refs {
		id, _ := ParseBranchRef(e.Content)
		out = append(out, BranchInfo{Label: fmt.Sprintf("B%d", i+1), RootID: id, Content: e.Content})
	}
	return out
}

// ResolveBranch finds a branch by the label a model would pass to read_branch (e.g. "B3") or,
// as a fallback for anything that still has a raw ID (older sessions, direct programmatic use),
// by that ID directly. ok is false if neither matches anything in log.
func ResolveBranch(log *SessionLog, labelOrID string) (rootID string, ok bool) {
	for _, b := range ListBranches(log) {
		if b.Label == labelOrID {
			return b.RootID, true
		}
	}
	if _, found := log.Get(labelOrID); found {
		return labelOrID, true
	}
	return "", false
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
