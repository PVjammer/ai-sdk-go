package agent

import (
	"context"

	"github.com/pvjammer/ai-sdk-go/pkg/function"
)

// AIAgent defines the interface for all agent implementations.
//
// Agents follow the same pattern as AIFunction: they are registered in a
// registry, discoverable by name, and configured via TOML. The AIFunction
// pattern maps to AIAgent as:
//
//	AIFunction.Execute()    → AIAgent.Run()
//	AIFunction.Metadata()  → AIAgent.Metadata()
//	AIFunction.Config()    → AIAgent.Config()
type AIAgent interface {
	// Metadata returns descriptive information about this agent.
	Metadata() *AgentMetadata

	// Config returns the agent's configuration.
	Config() function.Config

	// Run executes the agent on a single user message for a given session.
	//
	// Returns a channel of events that streams the agent's actions and responses.
	// The channel is closed when the agent completes or encounters an error.
	//
	// Multiple calls with the same sessionID share conversation history.
	Run(ctx context.Context, sessionID, message string) <-chan Event

	// RegisterTool adds a tool (AIFunction) to the agent's toolbox.
	RegisterTool(fn function.AIFunction) error

	// SetApprovalFunc installs a callback that is invoked before each tool execution.
	// If the callback returns false, the tool call is skipped.
	// If nil, all tools execute without approval.
	SetApprovalFunc(fn func(toolName string, input interface{}) (bool, error))

	// SetHooks installs lifecycle hooks into the agent loop.
	// Hooks fire at specific points during execution and can transform messages,
	// tool calls, and results. See AgentHooks for the full set of hook points
	// and their execution order.
	SetHooks(hooks AgentHooks)

	// GetHooks returns the currently installed AgentHooks.
	// Useful for composing additional hooks on top of the defaults installed
	// by the agent implementation (e.g. layering Logging in verbose mode).
	GetHooks() AgentHooks
}

// AgentMetadata contains descriptive information about an agent type.
type AgentMetadata struct {
	// Name is the human-readable agent name (e.g., "Simple Agent").
	Name string

	// Description explains what this agent does.
	Description string

	// CLIName is the short name used for config and CLI references (e.g., "simple").
	CLIName string
}
