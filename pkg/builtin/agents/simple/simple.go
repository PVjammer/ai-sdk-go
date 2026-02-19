// Package simple provides the default SimpleAgent built on native Ollama tool calling.
//
// Unlike the legacy planner-based agent that parsed USE_TOOL: patterns from raw text,
// SimpleAgent uses Ollama's structured tool API — tools are passed as JSON schemas
// and the model returns structured tool_calls with no parsing required.
package simple

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/pvjammer/ai-sdk-go/pkg/agent"
	"github.com/pvjammer/ai-sdk-go/pkg/function"
	"github.com/pvjammer/ai-sdk-go/pkg/llm"
)

const defaultSystemPrompt = "You are a helpful AI assistant. Use the available tools when they would help answer the user's question."

// SimpleAgent is the default agent that uses Ollama's native tool calling API.
//
// It embeds AgentBase for session management, tool registry, and the core
// executeAgentLoop. The LLM provider is resolved lazily from config on first Run().
type SimpleAgent struct {
	agent.AgentBase

	// Lazy provider initialization
	providerOnce sync.Once
	providerErr  error
}

// New creates a new SimpleAgent with the given config.
//
// The LLM provider is not created until the first Run() call, so that
// config bridging (BridgeToViper) has completed before we read LLM settings.
func New(cfg function.Config) agent.AIAgent {
	a := &SimpleAgent{}
	a.InitBase(cfg)
	a.StartAutoCleanup(1*time.Hour, 24*time.Hour)
	return a
}

// Metadata returns descriptive information about the SimpleAgent.
func (a *SimpleAgent) Metadata() *agent.AgentMetadata {
	return &agent.AgentMetadata{
		Name:        "Simple Agent",
		Description: "A helpful AI assistant with native Ollama tool calling support",
		CLIName:     "simple",
	}
}

// resolveProvider initializes the LLM provider from config (once, thread-safe).
func (a *SimpleAgent) resolveProvider() error {
	a.providerOnce.Do(func() {
		cfg := a.Config()

		endpoint := cfg.GetString("llm.ollama.endpoint")
		if endpoint == "" {
			endpoint = "http://localhost:11434"
		}

		model := cfg.GetString("llm.ollama.model")
		if model == "" {
			model = "llama3.2:latest"
		}

		provider, err := llm.NewOllamaProvider(endpoint, model)
		if err != nil {
			a.providerErr = fmt.Errorf("failed to create LLM provider: %w", err)
			return
		}

		a.SetProvider(provider)
	})
	return a.providerErr
}

// systemPrompt returns the system prompt, reading from config if set.
func (a *SimpleAgent) systemPrompt() string {
	if p := a.Config().GetString("simple.system_prompt"); p != "" {
		return p
	}
	return defaultSystemPrompt
}

// maxIterations returns the iteration limit from config (default 10).
func (a *SimpleAgent) maxIterations() int {
	if n := a.Config().GetInt("agent.max_iterations"); n > 0 {
		return n
	}
	return 10
}

// thinkingVisible returns whether to emit thinking events (default false).
func (a *SimpleAgent) thinkingVisible() bool {
	return a.Config().GetBool("simple.thinking_visible")
}

// Run executes the agent on a user message for the given session.
//
// It resolves the LLM provider on the first call, then delegates to
// AgentBase.BaseRun() which manages session locking and the tool loop.
func (a *SimpleAgent) Run(ctx context.Context, sessionID, message string) <-chan agent.Event {
	// Resolve provider before delegating — emit error if it fails
	if err := a.resolveProvider(); err != nil {
		events := make(chan agent.Event, 1)
		events <- agent.Event{
			Type:      agent.EventError,
			Data:      agent.ErrorEvent{Error: err, Message: "Failed to initialize LLM provider"},
			Timestamp: time.Now(),
		}
		close(events)
		return events
	}

	return a.BaseRun(ctx, sessionID, message, a.systemPrompt(), a.maxIterations(), a.thinkingVisible())
}
