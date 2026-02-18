package agent

import (
	"github.com/pvjammer/ai-sdk-go/pkg/agentcontext"
)

// AgentConfig configures agent behavior and capabilities.
type AgentConfig struct {
	// Name is a human-readable identifier for this agent
	Name string

	// Description explains what this agent does
	Description string

	// SystemPrompt is the base instruction for the agent
	SystemPrompt string

	// MaxIterations prevents infinite loops (default: 10)
	MaxIterations int

	// MaxTokens limits tokens per LLM call (default: 2048)
	MaxTokens int

	// Temperature controls LLM randomness (default: 0.7)
	Temperature float64

	// StreamResponses enables real-time text streaming (default: true)
	StreamResponses bool

	// ThinkingVisible shows internal reasoning to user (default: false)
	ThinkingVisible bool

	// Context is the AgentContext implementation to use
	// If nil, a new InMemory context will be created
	Context agentcontext.AgentContext

	// SessionCleanupInterval controls how often to check for old sessions (default: 1 hour)
	SessionCleanupInterval int64 // in seconds

	// SessionTimeout controls when inactive sessions are removed (default: 24 hours)
	SessionTimeout int64 // in seconds

	// ApprovalFunc is called before executing a tool when set.
	// It receives the tool name and input, and returns whether execution is approved.
	// If nil, all tools execute without approval.
	ApprovalFunc func(toolName string, input interface{}) (bool, error)
}

// DefaultConfig returns sensible defaults for most use cases.
func DefaultConfig() *AgentConfig {
	return &AgentConfig{
		Name:                   "Agent",
		Description:            "A helpful AI agent",
		SystemPrompt:           "You are a helpful AI assistant. Use tools when appropriate.",
		MaxIterations:          10,
		MaxTokens:              2048,
		Temperature:            0.7,
		StreamResponses:        true,
		ThinkingVisible:        false,
		Context:                nil, // Will be created as InMemory by default
		SessionCleanupInterval: 3600,  // 1 hour
		SessionTimeout:         86400, // 24 hours
	}
}

// Merge combines this config with another, with the other taking precedence.
func (c *AgentConfig) Merge(other *AgentConfig) *AgentConfig {
	if other == nil {
		return c
	}

	result := *c // Copy

	if other.Name != "" {
		result.Name = other.Name
	}
	if other.Description != "" {
		result.Description = other.Description
	}
	if other.SystemPrompt != "" {
		result.SystemPrompt = other.SystemPrompt
	}
	if other.MaxIterations > 0 {
		result.MaxIterations = other.MaxIterations
	}
	if other.MaxTokens > 0 {
		result.MaxTokens = other.MaxTokens
	}
	if other.Temperature > 0 {
		result.Temperature = other.Temperature
	}
	if other.Context != nil {
		result.Context = other.Context
	}
	if other.SessionCleanupInterval > 0 {
		result.SessionCleanupInterval = other.SessionCleanupInterval
	}
	if other.SessionTimeout > 0 {
		result.SessionTimeout = other.SessionTimeout
	}

	// Boolean fields - take from other regardless
	result.StreamResponses = other.StreamResponses
	result.ThinkingVisible = other.ThinkingVisible

	if other.ApprovalFunc != nil {
		result.ApprovalFunc = other.ApprovalFunc
	}

	return &result
}
