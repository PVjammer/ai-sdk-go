package workflow

import "context"

// LLMCallOptions configures an LLM call from within a workflow step.
type LLMCallOptions struct {
	// Model overrides the engine's default model.
	Model string

	// MaxTokens overrides the engine's default max_tokens.
	MaxTokens int

	// Temperature overrides the engine's default temperature.
	Temperature float64

	// SystemPrompt is the system-level instruction for the LLM.
	SystemPrompt string
}

// Executor is injected into each step runner, enabling testing without a real Engine.
//
// The Engine implements this interface; tests inject a mock implementation.
type Executor interface {
	// CallTool calls a registered AIFunction tool by name with the given input map.
	CallTool(ctx context.Context, name string, input map[string]interface{}) (interface{}, error)

	// CallLLM calls the configured LLM provider with the given prompt and options.
	CallLLM(ctx context.Context, prompt string, opts *LLMCallOptions) (string, error)

	// Confirm blocks until the consumer sends on the ConfirmEvent.ResponseCh
	// (true = approved, false = denied) or the context is cancelled.
	// Returns true if approved, false if denied, or an error on cancellation.
	Confirm(ctx context.Context, stepID, prompt string, ctxValue interface{}) (bool, error)

	// EmitEvent sends an event on the workflow's event channel.
	EmitEvent(ev Event)
}
