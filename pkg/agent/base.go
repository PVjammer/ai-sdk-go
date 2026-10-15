package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/pvjammer/ai-sdk-go/pkg/function"
	"github.com/pvjammer/ai-sdk-go/pkg/llm"
	"github.com/pvjammer/ai-sdk-go/pkg/skill"
)

// AgentBase provides shared implementation for all AI agent types.
//
// Concrete agents embed AgentBase and call InitBase() in their constructor,
// then implement the AIAgent interface by delegating Run() to BaseRun().
//
// Session history is stored per sessionID and persists across multiple Run() calls.
// The native Ollama tool-calling API is used — no regex parsing of LLM output.
type AgentBase struct {
	function.Base

	// provider is the LLM provider with native tool calling support.
	// Set via SetProvider() before the first Run() call.
	provider llm.ToolCallingProvider

	// Session storage: sessionID → agentSession
	sessionsMu sync.RWMutex
	sessions   map[string]*agentSession

	// Tool registry: tool name → AIFunction
	toolsMu sync.RWMutex
	tools   map[string]function.AIFunction

	// sequentialTools marks tool names that must not run concurrently with the rest of their
	// batch. See SetSequentialTools and docs/design/turn-branching.md's "Tool-call concurrency".
	sequentialTools map[string]bool

	// approvalFunc is called before each tool execution (optional).
	approvalFunc func(toolName string, input interface{}) (bool, error)

	// hooks contains optional lifecycle callbacks for the agent loop.
	hooks AgentHooks

	// skillLoader, when non-nil, appends the skill catalog XML to the system
	// prompt so the model knows which skills are available.
	skillLoader *skill.SkillLoader

	// Session cleanup
	cleanupTicker *time.Ticker
	cleanupDone   chan struct{}
}

// agentSession holds per-session conversation history and locking.
type agentSession struct {
	mu           sync.Mutex
	history      []llm.ChatMessage
	lastAccessed time.Time
}

// InitBase initializes AgentBase fields. Must be called by concrete agents in New().
func (b *AgentBase) InitBase(cfg function.Config) {
	b.Base = function.NewBase(cfg)
	b.sessions = make(map[string]*agentSession)
	b.tools = make(map[string]function.AIFunction)
}

// SetProvider sets the LLM provider for tool-calling requests.
func (b *AgentBase) SetProvider(provider llm.ToolCallingProvider) {
	b.provider = provider
}

// SetApprovalFunc installs a callback invoked before each tool execution.
func (b *AgentBase) SetApprovalFunc(fn func(toolName string, input interface{}) (bool, error)) {
	b.approvalFunc = fn
}

// SetHooks installs lifecycle hooks into the agent loop.
// Hooks fire at six points during execution; see AgentHooks for details.
func (b *AgentBase) SetHooks(hooks AgentHooks) {
	b.hooks = hooks
}

// GetHooks returns the currently installed AgentHooks.
func (b *AgentBase) GetHooks() AgentHooks {
	return b.hooks
}

// SetSkillLoader attaches a SkillLoader whose catalog XML is appended to the
// system prompt on every Run() call, making skills discoverable by the model.
func (b *AgentBase) SetSkillLoader(loader *skill.SkillLoader) {
	b.skillLoader = loader
}

// SetSequentialTools marks tool names whose batch must run sequentially rather than
// concurrently with the rest of that round's tool calls. Tool-call batches are parallel by
// default (docs/design/turn-branching.md); a batch containing ANY sequential-tagged tool runs
// the WHOLE batch sequentially, not just that one call — matching pi's model.
func (b *AgentBase) SetSequentialTools(names ...string) {
	b.toolsMu.Lock()
	defer b.toolsMu.Unlock()
	if b.sequentialTools == nil {
		b.sequentialTools = make(map[string]bool)
	}
	for _, n := range names {
		b.sequentialTools[n] = true
	}
}

func (b *AgentBase) batchIsSequential(calls []llm.ToolCall) bool {
	b.toolsMu.RLock()
	defer b.toolsMu.RUnlock()
	for _, tc := range calls {
		if b.sequentialTools[tc.Name] {
			return true
		}
	}
	return false
}

// RegisterTool adds an AIFunction as a tool available to the agent.
func (b *AgentBase) RegisterTool(fn function.AIFunction) error {
	name := fn.Metadata().GetName("tool")
	if name == "" {
		return fmt.Errorf("tool has no name")
	}

	b.toolsMu.Lock()
	defer b.toolsMu.Unlock()

	b.tools[name] = fn
	return nil
}

// toolsAsToolDefs converts the registered AIFunctions to llm.ToolDef slice
// for the native Ollama tool calling API.
func (b *AgentBase) toolsAsToolDefs() []llm.ToolDef {
	b.toolsMu.RLock()
	defer b.toolsMu.RUnlock()

	defs := make([]llm.ToolDef, 0, len(b.tools))
	for _, fn := range b.tools {
		meta := fn.Metadata()
		defs = append(defs, llm.ToolDef{
			Name:        meta.GetName("tool"),
			Description: meta.GetDescription("tool"),
			Parameters:  fn.InputSchema(),
		})
	}
	return defs
}

// executeTool runs the named tool with the given arguments.
func (b *AgentBase) executeTool(ctx context.Context, call llm.ToolCall) (interface{}, error) {
	b.toolsMu.RLock()
	fn, ok := b.tools[call.Name]
	b.toolsMu.RUnlock()

	if !ok {
		return nil, fmt.Errorf("%w: %s", ErrToolNotFound, call.Name)
	}

	return fn.Execute(ctx, call.Args)
}

// getOrCreateSession retrieves an existing session or creates a new one.
func (b *AgentBase) getOrCreateSession(sessionID string) *agentSession {
	b.sessionsMu.RLock()
	if s, ok := b.sessions[sessionID]; ok {
		b.sessionsMu.RUnlock()
		return s
	}
	b.sessionsMu.RUnlock()

	b.sessionsMu.Lock()
	defer b.sessionsMu.Unlock()

	// Double-check after acquiring write lock
	if s, ok := b.sessions[sessionID]; ok {
		return s
	}

	s := &agentSession{
		history:      make([]llm.ChatMessage, 0),
		lastAccessed: time.Now(),
	}
	b.sessions[sessionID] = s
	return s
}

// ClearSession removes a session's conversation history.
func (b *AgentBase) ClearSession(sessionID string) {
	b.sessionsMu.Lock()
	defer b.sessionsMu.Unlock()
	delete(b.sessions, sessionID)
}

// SeedSession creates (or replaces) a session with the given initial history,
// without running a turn. Used by Session/turn-branching (docs/design/turn-branching.md) to
// prime a scratch session with a path through a SessionLog before running a turn against it —
// the scratch session is otherwise an ordinary AgentBase session and sees normal hook
// processing on its next Run().
func (b *AgentBase) SeedSession(sessionID string, history []llm.ChatMessage) {
	b.sessionsMu.Lock()
	defer b.sessionsMu.Unlock()

	seeded := make([]llm.ChatMessage, len(history))
	copy(seeded, history)
	b.sessions[sessionID] = &agentSession{
		history:      seeded,
		lastAccessed: time.Now(),
	}
}

// SessionHistory returns a copy of a session's current message history. Used by
// Session/turn-branching to read back what a scratch session accumulated after a turn, before
// the scratch session is discarded via ClearSession.
func (b *AgentBase) SessionHistory(sessionID string) []llm.ChatMessage {
	b.sessionsMu.RLock()
	defer b.sessionsMu.RUnlock()

	s, ok := b.sessions[sessionID]
	if !ok {
		return nil
	}
	out := make([]llm.ChatMessage, len(s.history))
	copy(out, s.history)
	return out
}

// StartAutoCleanup starts a background goroutine to remove stale sessions.
func (b *AgentBase) StartAutoCleanup(interval, timeout time.Duration) {
	b.cleanupTicker = time.NewTicker(interval)
	b.cleanupDone = make(chan struct{})

	go func() {
		for {
			select {
			case <-b.cleanupTicker.C:
				b.cleanupOldSessions(timeout)
			case <-b.cleanupDone:
				return
			}
		}
	}()
}

func (b *AgentBase) cleanupOldSessions(timeout time.Duration) {
	b.sessionsMu.Lock()
	defer b.sessionsMu.Unlock()

	now := time.Now()
	for id, s := range b.sessions {
		if now.Sub(s.lastAccessed) > timeout {
			delete(b.sessions, id)
		}
	}
}

// Close stops the session cleanup goroutine.
func (b *AgentBase) Close() {
	if b.cleanupTicker != nil {
		b.cleanupTicker.Stop()
		close(b.cleanupDone)
	}
}

// BaseRun is the template method for executing the agent loop.
//
// Concrete agents call BaseRun from their Run() implementation after
// resolving the LLM provider and reading config values.
//
// It handles session locking, history management, and event emission.
func (b *AgentBase) BaseRun(
	ctx context.Context,
	sessionID, message string,
	systemPrompt string,
	maxIterations int,
	thinkingVisible bool,
) <-chan Event {
	// Append skill catalog XML to system prompt when a loader is attached.
	// This follows the Claude Code pattern: only name+description in context by default.
	if b.skillLoader != nil {
		if catalog := b.skillLoader.CatalogXML(); catalog != "" {
			systemPrompt = systemPrompt + "\n\n" + catalog
		}
	}

	events := make(chan Event, 50)

	go func() {
		defer close(events)

		// Get or create session
		session := b.getOrCreateSession(sessionID)

		// Prevent concurrent execution on the same session
		if !session.mu.TryLock() {
			events <- Event{
				Type:      EventError,
				Data:      ErrorEvent{Error: ErrConcurrentExecution, Message: "Session is already executing"},
				Timestamp: time.Now(),
			}
			return
		}
		defer session.mu.Unlock()

		session.lastAccessed = time.Now()

		// Run the agent loop
		b.executeAgentLoop(ctx, sessionID, session, message, systemPrompt, maxIterations, thinkingVisible, events)
	}()

	return events
}

// executeAgentLoop is the core agent loop using native Ollama tool calling.
//
// Hook call sites (in order):
//  1. PreInput   — transforms user message before history append
//  2. PreModel   — transforms message slice before each LLM call
//  3. PostModel  — transforms LLM response before tool dispatch
//  4. [EventToolCall emitted]
//  5. PreTool    — can modify args or deny execution
//  6. [approvalFunc — existing backward-compat mechanism]
//  7. executeTool
//  8. [EventToolResult emitted with raw output]
//  9. PostTool   — transforms result string before history append
//
// 10. PostResponse — called with final text response (non-fatal errors)
func (b *AgentBase) executeAgentLoop(
	ctx context.Context,
	sessionID string,
	session *agentSession,
	message string,
	systemPrompt string,
	maxIterations int,
	thinkingVisible bool,
	events chan<- Event,
) {
	// Inject session ID into context so tools (e.g. activate_skill) can read it.
	ctx = skill.WithSessionID(ctx, sessionID)

	// ── Hook: PreInput ──────────────────────────────────────────────────────
	if b.hooks.PreInput != nil {
		var err error
		message, err = b.hooks.PreInput(ctx, sessionID, message)
		if err != nil {
			events <- Event{
				Type:      EventError,
				Data:      ErrorEvent{Error: err, Message: "PreInput hook failed"},
				Timestamp: time.Now(),
			}
			return
		}
	}

	// Append user message to persistent history
	session.history = append(session.history, llm.ChatMessage{
		Role:    "user",
		Content: message,
	})

	toolDefs := b.toolsAsToolDefs()
	opts := llm.DefaultOptions()
	opts.Stream = false // Tool calls must be complete before execution

	for i := 0; i < maxIterations; i++ {
		if ctx.Err() != nil {
			events <- Event{
				Type:      EventError,
				Data:      ErrorEvent{Error: ctx.Err(), Message: "Context cancelled"},
				Timestamp: time.Now(),
			}
			return
		}

		// Build full message list: optional system + session history
		messages := make([]llm.ChatMessage, 0, len(session.history)+1)
		if systemPrompt != "" {
			messages = append(messages, llm.ChatMessage{
				Role:    "system",
				Content: systemPrompt,
			})
		}
		messages = append(messages, session.history...)

		// ── Hook: PreModel ─────────────────────────────────────────────────
		if b.hooks.PreModel != nil {
			var err error
			messages, err = b.hooks.PreModel(ctx, messages)
			if err != nil {
				events <- Event{
					Type:      EventError,
					Data:      ErrorEvent{Error: err, Message: "PreModel hook failed"},
					Timestamp: time.Now(),
				}
				return
			}
		}

		// Call LLM with native tool calling
		chunkCh := b.provider.ChatWithTools(ctx, messages, toolDefs, opts)

		// Collect the final chunk (non-streaming, so one chunk expected)
		var finalChunk llm.StreamChunk
		for chunk := range chunkCh {
			if chunk.Error != nil {
				events <- Event{
					Type:      EventError,
					Data:      ErrorEvent{Error: chunk.Error, Message: "LLM error"},
					Timestamp: time.Now(),
				}
				return
			}
			if chunk.Done {
				finalChunk = chunk
			}
		}

		// ── Hook: PostModel ────────────────────────────────────────────────
		if b.hooks.PostModel != nil {
			var err error
			finalChunk, err = b.hooks.PostModel(ctx, finalChunk)
			if err != nil {
				events <- Event{
					Type:      EventError,
					Data:      ErrorEvent{Error: err, Message: "PostModel hook failed"},
					Timestamp: time.Now(),
				}
				return
			}
		}

		// Optionally show reasoning if the model returned text alongside tool calls
		if thinkingVisible && finalChunk.Text != "" && len(finalChunk.ToolCalls) > 0 {
			events <- Event{
				Type:      EventThinking,
				Data:      ThinkingEvent{Content: finalChunk.Text},
				Timestamp: time.Now(),
			}
		}

		// Handle tool calls
		if len(finalChunk.ToolCalls) > 0 {
			// Add the assistant's tool-call request to history
			session.history = append(session.history, llm.ChatMessage{
				Role:      "assistant",
				Content:   finalChunk.Text,
				ToolCalls: finalChunk.ToolCalls,
			})

			calls := finalChunk.ToolCalls

			// Announce every call in request order before dispatch, regardless of
			// execution mode, so an observer always learns what was requested in the
			// order the model asked for it.
			for _, tc := range calls {
				events <- Event{
					Type:      EventToolCall,
					Data:      ToolCallEvent{ToolName: tc.Name, Input: tc.Args},
					Timestamp: time.Now(),
				}
			}

			// Tool calls run in parallel by default (docs/design/turn-branching.md): every
			// call in the batch executes concurrently, EventToolResult fires in whatever
			// order calls actually finish, but results are written into history in the
			// assistant's ORIGINAL REQUEST ORDER below — the model never sees results out
			// of order even though they may have finished out of order. A batch containing
			// any tool registered via SetSequentialTools runs entirely sequentially instead.
			results := make([]llm.ChatMessage, len(calls))
			if b.batchIsSequential(calls) {
				for i, tc := range calls {
					results[i] = b.runToolCall(ctx, tc, events)
				}
			} else {
				var wg sync.WaitGroup
				wg.Add(len(calls))
				for i, tc := range calls {
					go func(i int, tc llm.ToolCall) {
						defer wg.Done()
						results[i] = b.runToolCall(ctx, tc, events)
					}(i, tc)
				}
				wg.Wait()
			}

			for _, m := range results {
				session.history = append(session.history, m)
			}

			// Continue loop: let the model respond now that it has tool results
			continue
		}

		// No tool calls — this is the final text response
		if finalChunk.Text != "" {
			session.history = append(session.history, llm.ChatMessage{
				Role:    "assistant",
				Content: finalChunk.Text,
			})

			// ── Hook: PostResponse (non-fatal) ─────────────────────────────
			if b.hooks.PostResponse != nil {
				if err := b.hooks.PostResponse(ctx, finalChunk.Text); err != nil {
					events <- Event{
						Type:      EventError,
						Data:      ErrorEvent{Error: err, Message: "PostResponse hook failed"},
						Timestamp: time.Now(),
					}
					// Non-fatal: fall through to emit EventResponse + EventComplete
				}
			}

			events <- Event{
				Type:      EventResponse,
				Data:      ResponseEvent{Content: finalChunk.Text, Done: true},
				Timestamp: time.Now(),
			}
			events <- Event{
				Type:      EventComplete,
				Data:      nil,
				Timestamp: time.Now(),
			}
			return
		}

		// Empty response with no tool calls — something went wrong
		events <- Event{
			Type:      EventError,
			Data:      ErrorEvent{Error: fmt.Errorf("LLM returned empty response"), Message: "Empty response from LLM"},
			Timestamp: time.Now(),
		}
		return
	}

	// Max iterations reached
	events <- Event{
		Type: EventError,
		Data: ErrorEvent{
			Error:   ErrMaxIterations,
			Message: fmt.Sprintf("Agent exceeded maximum iterations (%d)", maxIterations),
		},
		Timestamp: time.Now(),
	}
}

// runToolCall executes one tool call end to end — PreTool, approval, execution, PostTool — and
// emits EventToolResult itself (EventToolCall has already been announced by the caller, in
// request order, before any call in the batch is dispatched). Safe to run concurrently with
// other calls from the same batch: it only touches its own tc/events parameters and the
// session's tool registry/hooks, which are read-only from here (events is a buffered channel,
// safe for concurrent sends; the caller writes the returned message into session.history itself,
// serially, after every call in the batch has finished).
func (b *AgentBase) runToolCall(ctx context.Context, tc llm.ToolCall, events chan<- Event) llm.ChatMessage {
	// ── Hook: PreTool ──────────────────────────────────────────────────────
	// Can modify args or silently deny.
	if b.hooks.PreTool != nil {
		modifiedCall, proceed, err := b.hooks.PreTool(ctx, tc)
		if err != nil {
			events <- Event{
				Type:      EventToolResult,
				Data:      ToolResultEvent{ToolName: tc.Name, Error: fmt.Errorf("PreTool hook error: %w", err)},
				Timestamp: time.Now(),
			}
			return llm.ChatMessage{
				Role:       "tool",
				Content:    fmt.Sprintf("error: PreTool hook failed for %s: %v", tc.Name, err),
				ToolCallID: tc.ID,
				ToolName:   tc.Name,
			}
		}
		if !proceed {
			events <- Event{
				Type:      EventToolResult,
				Data:      ToolResultEvent{ToolName: tc.Name, Error: fmt.Errorf("tool execution denied by PreTool hook")},
				Timestamp: time.Now(),
			}
			return llm.ChatMessage{
				Role:       "tool",
				Content:    fmt.Sprintf("denied: %s was blocked by a hook", tc.Name),
				ToolCallID: tc.ID,
				ToolName:   tc.Name,
			}
		}
		tc = modifiedCall // use the (possibly modified) call going forward
	}

	// ── approvalFunc (backward-compat; runs after PreTool) ─────────────────
	if b.approvalFunc != nil {
		approved, approvalErr := b.approvalFunc(tc.Name, tc.Args)
		if approvalErr != nil {
			events <- Event{
				Type:      EventToolResult,
				Data:      ToolResultEvent{ToolName: tc.Name, Error: fmt.Errorf("approval error: %w", approvalErr)},
				Timestamp: time.Now(),
			}
			return llm.ChatMessage{
				Role:       "tool",
				Content:    fmt.Sprintf("error: approval failed for %s", tc.Name),
				ToolCallID: tc.ID,
				ToolName:   tc.Name,
			}
		}
		if !approved {
			events <- Event{
				Type:      EventToolResult,
				Data:      ToolResultEvent{ToolName: tc.Name, Error: fmt.Errorf("tool execution denied by user")},
				Timestamp: time.Now(),
			}
			return llm.ChatMessage{
				Role:       "tool",
				Content:    fmt.Sprintf("denied: user did not approve %s", tc.Name),
				ToolCallID: tc.ID,
				ToolName:   tc.Name,
			}
		}
	}

	// Execute the tool
	output, execErr := b.executeTool(ctx, tc)

	// ── EventToolResult (raw output — UI sees the real result) ─────────────
	events <- Event{
		Type:      EventToolResult,
		Data:      ToolResultEvent{ToolName: tc.Name, Output: output, Error: execErr},
		Timestamp: time.Now(),
	}

	// Marshal result to string for history
	var resultContent string
	if execErr != nil {
		resultContent = fmt.Sprintf("error: %v", execErr)
	} else {
		resultBytes, marshalErr := json.Marshal(output)
		if marshalErr != nil {
			resultContent = fmt.Sprintf("%v", output)
		} else {
			resultContent = string(resultBytes)
		}
	}

	// ── Hook: PostTool ──────────────────────────────────────────────────────
	// Transforms result string before it enters history. The model sees the hook's
	// output; the UI already saw the raw output via EventToolResult above.
	if b.hooks.PostTool != nil {
		modified, hookErr := b.hooks.PostTool(ctx, tc.Name, resultContent, execErr)
		if hookErr != nil {
			resultContent = fmt.Sprintf("error: PostTool hook failed: %v", hookErr)
		} else {
			resultContent = modified
		}
	}

	return llm.ChatMessage{
		Role:       "tool",
		Content:    resultContent,
		ToolCallID: tc.ID,
		ToolName:   tc.Name,
	}
}
