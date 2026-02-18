// Package agentcontext provides the core context abstraction for agent conversations.
//
// AgentContext is the central abstraction that manages conversation history,
// personalization, context switching, and compression for local LLM agents.
package agentcontext

import (
	"time"
)

// AgentContext is the core abstraction for managing agent conversations.
//
// It provides three layers of functionality:
//  1. Core: Conversation management (always present)
//  2. Extended: Personalization & state (optional, pluggable)
//  3. Managed: Length handling (automatic via hooks)
type AgentContext interface {
	// Core: Conversation management (always present)
	AddMessage(msg Message)
	GetMessages(limit int) []Message
	Clear()

	// Extended: Personalization & state (optional, pluggable)
	SetPersonalization(key string, value interface{})
	GetPersonalization(key string) interface{}

	// Switching: Multiple contexts (optional, pluggable)
	SwitchContext(contextID string) error
	GetCurrentContextID() string

	// Managed: Length handling (automatic via hooks)
	EstimateTokens() int
	NeedsCompression() bool
	RegisterCompressionHook(hook CompressionHook)
}

// Message represents a single message in the conversation.
type Message struct {
	Role      string                 // "user", "assistant", "system", "tool"
	Content   string                 // Message content
	ToolCalls []ToolCall             // Tool calls made in this message
	Metadata  map[string]interface{} // Extensibility for "other things"
	Timestamp time.Time              // When the message was created
}

// ToolCall represents a tool invocation within a message.
type ToolCall struct {
	ID     string      // Unique identifier for this tool call
	Name   string      // Name of the tool being called
	Input  interface{} // Input parameters for the tool
	Output interface{} // Output from the tool (nil if not yet executed)
	Error  error       // Error if tool execution failed
}

// CompressionHook is a pluggable strategy for managing context length.
//
// Multiple hooks can be registered and they execute in priority order
// (higher priority first) when NeedsCompression() returns true.
type CompressionHook interface {
	// ShouldCompress determines if this hook should run based on context state
	ShouldCompress(agentCtx AgentContext) bool

	// Compress applies the compression strategy and returns modified context
	Compress(agentCtx AgentContext) (AgentContext, error)

	// Priority determines execution order (higher values run first)
	Priority() int
}
