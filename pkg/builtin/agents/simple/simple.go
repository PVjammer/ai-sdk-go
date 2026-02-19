// Package simple provides the default SimpleAgent built on native Ollama tool calling.
//
// Unlike the legacy planner-based agent that parsed USE_TOOL: patterns from raw text,
// SimpleAgent uses Ollama's structured tool API — tools are passed as JSON schemas
// and the model returns structured tool_calls with no parsing required.
//
// # Default hooks
//
// SimpleAgent installs three hooks automatically based on config:
//
//   - TokenLimit (PreModel): trims oldest conversation exchanges before each LLM call
//     when estimated token count exceeds agent.context_tokens (default: 8000).
//
//   - LargeResult (PostTool): redirects tool results larger than
//     agent.large_result_threshold bytes (default: 8192) to a temp file,
//     replacing the history entry with a short reference the model can read_file.
//
//   - Logging (all points): logs every hook point to stderr when
//     agent.debug_hooks is true (default: false).
//
// All three can be overridden by calling SetHooks() after construction.
package simple

import (
	"context"
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/pvjammer/ai-sdk-go/pkg/agent"
	agenthooks "github.com/pvjammer/ai-sdk-go/pkg/agent/hooks"
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
// Hooks are installed immediately since they only need config values, not the LLM.
func New(cfg function.Config) agent.AIAgent {
	a := &SimpleAgent{}
	a.InitBase(cfg)
	a.StartAutoCleanup(1*time.Hour, 24*time.Hour)

	// Install default hooks. Callers can override with SetHooks() after New().
	a.SetHooks(buildHooks(cfg))

	return a
}

// buildHooks constructs the default AgentHooks for SimpleAgent from config.
//
// Hook configuration keys (all under the [agent] section in config.toml):
//
//	agent.context_tokens         int  (default 8000)  — token limit for PreModel trimming
//	agent.large_result_threshold int  (default 8192)  — byte threshold for PostTool file redirect
//	agent.debug_hooks            bool (default false) — enable Logging hook to stderr
func buildHooks(cfg function.Config) agent.AgentHooks {
	var hs []agent.AgentHooks

	// ── Context window management ─────────────────────────────────────────
	// Trim oldest conversation exchanges before each LLM call to stay within
	// the model's context window. The heuristic is 4 chars/token.
	contextTokens := cfg.GetInt("agent.context_tokens")
	if contextTokens <= 0 {
		contextTokens = 8000
	}
	hs = append(hs, agenthooks.TokenLimit(contextTokens))

	// ── Large result handling ─────────────────────────────────────────────
	// Redirect oversized tool results to temp files so they don't blow out
	// the context window. The model receives a file path it can read_file.
	largeResultThreshold := cfg.GetInt("agent.large_result_threshold")
	if largeResultThreshold <= 0 {
		largeResultThreshold = 8192
	}
	hs = append(hs, agenthooks.LargeResult(largeResultThreshold, ""))

	// ── Optional debug logging ────────────────────────────────────────────
	// Logs all six hook points to stderr. Useful during development.
	// Enable with: agent.debug_hooks = true in config.toml
	if cfg.GetBool("agent.debug_hooks") {
		hs = append(hs, agenthooks.Logging(os.Stderr))
	}

	return agenthooks.Combine(hs...)
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

// systemPrompt returns the system prompt with current UTC time appended.
func (a *SimpleAgent) systemPrompt() string {
	base := a.Config().GetString("simple.system_prompt")
	if base == "" {
		base = defaultSystemPrompt
	}
	now := time.Now().UTC()
	return base + fmt.Sprintf("\n\nCurrent date and time: %s UTC", now.Format("2006-01-02 15:04:05"))
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
