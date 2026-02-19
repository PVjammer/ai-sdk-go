package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/pvjammer/ai-sdk-go/pkg/function"
	"github.com/pvjammer/ai-sdk-go/pkg/llm"
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

	// approvalFunc is called before each tool execution (optional).
	approvalFunc func(toolName string, input interface{}) (bool, error)

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
		b.executeAgentLoop(ctx, session, message, systemPrompt, maxIterations, thinkingVisible, events)
	}()

	return events
}

// executeAgentLoop is the core agent loop using native Ollama tool calling.
//
// Loop:
//  1. Append user message to session history
//  2. For up to maxIterations:
//     a. Build messages = [system] + session history
//     b. Call LLM with tool schemas via ChatWithTools
//     c. If response has ToolCalls: execute each tool, append results to history, continue
//     d. If response has Text (no ToolCalls): append to history, emit EventResponse + EventComplete, return
//  3. If max iterations reached: emit EventError
func (b *AgentBase) executeAgentLoop(
	ctx context.Context,
	session *agentSession,
	message string,
	systemPrompt string,
	maxIterations int,
	thinkingVisible bool,
	events chan<- Event,
) {
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

			// Execute each tool call
			for _, tc := range finalChunk.ToolCalls {
				events <- Event{
					Type:      EventToolCall,
					Data:      ToolCallEvent{ToolName: tc.Name, Input: tc.Args},
					Timestamp: time.Now(),
				}

				// Check approval if configured
				if b.approvalFunc != nil {
					approved, approvalErr := b.approvalFunc(tc.Name, tc.Args)
					if approvalErr != nil {
						events <- Event{
							Type:      EventToolResult,
							Data:      ToolResultEvent{ToolName: tc.Name, Error: fmt.Errorf("approval error: %w", approvalErr)},
							Timestamp: time.Now(),
						}
						session.history = append(session.history, llm.ChatMessage{
							Role:       "tool",
							Content:    fmt.Sprintf("error: approval failed for %s", tc.Name),
							ToolCallID: tc.ID,
							ToolName:   tc.Name,
						})
						continue
					}
					if !approved {
						events <- Event{
							Type:      EventToolResult,
							Data:      ToolResultEvent{ToolName: tc.Name, Error: fmt.Errorf("tool execution denied by user")},
							Timestamp: time.Now(),
						}
						session.history = append(session.history, llm.ChatMessage{
							Role:       "tool",
							Content:    fmt.Sprintf("denied: user did not approve %s", tc.Name),
							ToolCallID: tc.ID,
							ToolName:   tc.Name,
						})
						continue
					}
				}

				// Execute the tool
				output, err := b.executeTool(ctx, tc)

				events <- Event{
					Type:      EventToolResult,
					Data:      ToolResultEvent{ToolName: tc.Name, Output: output, Error: err},
					Timestamp: time.Now(),
				}

				// Format result content for history
				var resultContent string
				if err != nil {
					resultContent = fmt.Sprintf("error: %v", err)
				} else {
					resultBytes, marshalErr := json.Marshal(output)
					if marshalErr != nil {
						resultContent = fmt.Sprintf("%v", output)
					} else {
						resultContent = string(resultBytes)
					}
				}

				// Add tool result to history
				session.history = append(session.history, llm.ChatMessage{
					Role:       "tool",
					Content:    resultContent,
					ToolCallID: tc.ID,
					ToolName:   tc.Name,
				})
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
