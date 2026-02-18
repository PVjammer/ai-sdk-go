package agent

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/pvjammer/ai-sdk-go/pkg/agentcontext"
	"github.com/pvjammer/ai-sdk-go/pkg/function"
	"github.com/pvjammer/ai-sdk-go/pkg/llm"
	"github.com/pvjammer/ai-sdk-go/pkg/planner"
	"github.com/pvjammer/ai-sdk-go/pkg/tools"
)

// Agent orchestrates LLM interactions, tool usage, and conversation management.
//
// It supports multiple concurrent sessions with per-session mutex locks
// to prevent race conditions when the same session is accessed concurrently.
type Agent struct {
	llm           llm.Provider
	sessions      sync.Map // sessionID -> *AgentSession
	toolManager   *tools.ToolManager
	plannerFunc   func() planner.Planner // Factory for creating planners
	config        *AgentConfig
	cleanupTicker *time.Ticker
	cleanupDone   chan struct{}
}

// AgentSession represents a single conversation session.
//
// Each session has its own context and mutex lock to prevent
// concurrent execution on the same session.
type AgentSession struct {
	mu           sync.Mutex
	agentContext agentcontext.AgentContext
	lastAccessed time.Time
}

// Errors
var (
	ErrConcurrentExecution = fmt.Errorf("session is already executing, concurrent execution not allowed")
	ErrMaxIterations       = fmt.Errorf("agent exceeded maximum iterations")
	ErrToolNotFound        = fmt.Errorf("tool not found")
)

// New creates a new agent with the given LLM provider and config.
func New(provider llm.Provider, config *AgentConfig) *Agent {
	if config == nil {
		config = DefaultConfig()
	} else {
		config = DefaultConfig().Merge(config)
	}

	a := &Agent{
		llm:         provider,
		sessions:    sync.Map{},
		toolManager: tools.NewToolManager(),
		plannerFunc: func() planner.Planner {
			return planner.NewSimplePlanner(provider, config.SystemPrompt)
		},
		config: config,
	}

	// Start automatic session cleanup
	a.startAutoCleanup(
		time.Duration(config.SessionCleanupInterval)*time.Second,
		time.Duration(config.SessionTimeout)*time.Second,
	)

	return a
}

// RegisterTool adds a tool to the agent's toolbox.
func (a *Agent) RegisterTool(fn function.AIFunction) error {
	return a.toolManager.Register(fn)
}

// Run executes the agent on a single user message for a given session.
//
// Returns a channel of events that streams the agent's actions and responses.
// The channel is closed when the agent completes or encounters an error.
//
// Example:
//
//	events := agent.Run(ctx, "session123", "What's the weather in SF?")
//	for event := range events {
//	    switch event.Type {
//	    case EventResponse:
//	        fmt.Print(event.Data.(ResponseEvent).Content)
//	    case EventError:
//	        fmt.Println("Error:", event.Data)
//	    }
//	}
func (a *Agent) Run(ctx context.Context, sessionID, message string) <-chan Event {
	events := make(chan Event, 50)

	// Get or create session
	session := a.getOrCreateSession(sessionID)

	// Try to acquire session lock (prevents concurrent execution)
	if !session.mu.TryLock() {
		// Session is already executing
		go func() {
			defer close(events)
			events <- Event{
				Type: EventError,
				Data: ErrorEvent{
					Error:   ErrConcurrentExecution,
					Message: "This session is already processing a message",
				},
				Timestamp: time.Now(),
			}
		}()
		return events
	}

	// Execute in background
	go func() {
		defer close(events)
		defer session.mu.Unlock()

		// Update last accessed time
		session.lastAccessed = time.Now()

		// Run the agent loop
		a.executeAgentLoop(ctx, session, message, events)
	}()

	return events
}

// executeAgentLoop implements the core agent execution: Plan → Act → Repeat
func (a *Agent) executeAgentLoop(ctx context.Context, session *AgentSession, message string, events chan<- Event) {
	agentCtx := session.agentContext

	// Add user message to context
	agentCtx.AddMessage(agentcontext.Message{
		Role:      "user",
		Content:   message,
		Timestamp: time.Now(),
	})

	// Create planner for this execution
	plnr := a.plannerFunc()

	// Agent loop: Plan → Act → Repeat
	for i := 0; i < a.config.MaxIterations; i++ {
		// Check context cancellation
		if ctx.Err() != nil {
			events <- Event{
				Type: EventError,
				Data: ErrorEvent{
					Error:   ctx.Err(),
					Message: "Context cancelled",
				},
				Timestamp: time.Now(),
			}
			return
		}

		// Get recent messages for planning
		messages := agentCtx.GetMessages(10)

		// Get tool definitions
		toolDefs := a.toolManager.GetToolDefinitions()

		// Plan next action
		plan, err := plnr.Plan(ctx, messages, toolDefs)
		if err != nil {
			events <- Event{
				Type: EventError,
				Data: ErrorEvent{
					Error:   err,
					Message: "Planning failed",
				},
				Timestamp: time.Now(),
			}
			return
		}

		// Show thinking if enabled
		if a.config.ThinkingVisible && plan.Reasoning != "" {
			events <- Event{
				Type: EventThinking,
				Data: ThinkingEvent{
					Content: plan.Reasoning,
				},
				Timestamp: time.Now(),
			}
		}

		// Execute action based on plan
		switch plan.Action {
		case planner.ActionUseTool:
			// Execute all tool calls
			for idx := range plan.ToolCalls {
				tc := &plan.ToolCalls[idx]

				// Emit tool call event
				events <- Event{
					Type: EventToolCall,
					Data: ToolCallEvent{
						ToolName: tc.Name,
						Input:    tc.Input,
					},
					Timestamp: time.Now(),
				}

				// Check approval if configured
				if a.config.ApprovalFunc != nil {
					approved, approvalErr := a.config.ApprovalFunc(tc.Name, tc.Input)
					if approvalErr != nil {
						events <- Event{
							Type: EventToolResult,
							Data: ToolResultEvent{
								ToolName: tc.Name,
								Error:    fmt.Errorf("approval error: %w", approvalErr),
							},
							Timestamp: time.Now(),
						}
						continue
					}
					if !approved {
						tc.Error = fmt.Errorf("tool execution denied by user")
						events <- Event{
							Type: EventToolResult,
							Data: ToolResultEvent{
								ToolName: tc.Name,
								Error:    tc.Error,
							},
							Timestamp: time.Now(),
						}
						agentCtx.AddMessage(agentcontext.Message{
							Role:      "assistant",
							Content:   fmt.Sprintf("Tool %s was denied by user", tc.Name),
							ToolCalls: []agentcontext.ToolCall{*tc},
							Timestamp: time.Now(),
						})
						continue
					}
				}

				// Execute tool
				output, err := a.toolManager.Execute(ctx, tc.Name, tc.Input)

				// Update tool call with results
				tc.Output = output
				tc.Error = err

				// Emit tool result event
				events <- Event{
					Type: EventToolResult,
					Data: ToolResultEvent{
						ToolName: tc.Name,
						Output:   output,
						Error:    err,
					},
					Timestamp: time.Now(),
				}

				// Add tool call to context
				agentCtx.AddMessage(agentcontext.Message{
					Role:      "assistant",
					Content:   fmt.Sprintf("Used tool: %s", tc.Name),
					ToolCalls: []agentcontext.ToolCall{*tc},
					Timestamp: time.Now(),
				})
			}

			// Continue loop to plan next action

		case planner.ActionRespond:
			// Stream response if enabled
			if a.config.StreamResponses {
				// For now, send as single chunk
				// TODO: Implement actual streaming from LLM
				events <- Event{
					Type: EventResponse,
					Data: ResponseEvent{
						Content: plan.Response,
						Done:    false,
					},
					Timestamp: time.Now(),
				}
			}

			// Final response
			events <- Event{
				Type: EventResponse,
				Data: ResponseEvent{
					Content: plan.Response,
					Done:    true,
				},
				Timestamp: time.Now(),
			}

			// Add assistant response to context
			agentCtx.AddMessage(agentcontext.Message{
				Role:      "assistant",
				Content:   plan.Response,
				Timestamp: time.Now(),
			})

			// Done!
			events <- Event{
				Type:      EventComplete,
				Data:      nil,
				Timestamp: time.Now(),
			}
			return
		}
	}

	// Max iterations reached
	events <- Event{
		Type: EventError,
		Data: ErrorEvent{
			Error:   ErrMaxIterations,
			Message: fmt.Sprintf("Agent exceeded maximum iterations (%d)", a.config.MaxIterations),
		},
		Timestamp: time.Now(),
	}
}

// getOrCreateSession retrieves an existing session or creates a new one.
func (a *Agent) getOrCreateSession(sessionID string) *AgentSession {
	// Try to load existing session
	if value, ok := a.sessions.Load(sessionID); ok {
		return value.(*AgentSession)
	}

	// Create new session
	agentCtx := a.config.Context
	if agentCtx == nil {
		agentCtx = agentcontext.NewInMemory()
	}

	session := &AgentSession{
		agentContext: agentCtx,
		lastAccessed: time.Now(),
	}

	// Store and return
	actual, _ := a.sessions.LoadOrStore(sessionID, session)
	return actual.(*AgentSession)
}

// startAutoCleanup starts a background goroutine to clean up old sessions.
func (a *Agent) startAutoCleanup(interval, timeout time.Duration) {
	a.cleanupTicker = time.NewTicker(interval)
	a.cleanupDone = make(chan struct{})

	go func() {
		for {
			select {
			case <-a.cleanupTicker.C:
				a.cleanupOldSessions(timeout)
			case <-a.cleanupDone:
				return
			}
		}
	}()
}

// cleanupOldSessions removes sessions that haven't been accessed recently.
func (a *Agent) cleanupOldSessions(timeout time.Duration) {
	now := time.Now()
	a.sessions.Range(func(key, value interface{}) bool {
		session := value.(*AgentSession)
		if now.Sub(session.lastAccessed) > timeout {
			a.sessions.Delete(key)
		}
		return true
	})
}

// Close stops the agent and cleans up resources.
func (a *Agent) Close() {
	if a.cleanupTicker != nil {
		a.cleanupTicker.Stop()
		close(a.cleanupDone)
	}
}

// GetSession returns the AgentContext for a specific session (for inspection).
func (a *Agent) GetSession(sessionID string) (agentcontext.AgentContext, bool) {
	if value, ok := a.sessions.Load(sessionID); ok {
		session := value.(*AgentSession)
		return session.agentContext, true
	}
	return nil, false
}
