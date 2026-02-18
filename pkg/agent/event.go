// Package agent provides the core agent execution engine.
package agent

import (
	"time"
)

// EventType identifies the kind of event being emitted.
type EventType string

const (
	// EventThinking indicates the agent is reasoning about what to do next
	EventThinking EventType = "thinking"

	// EventToolCall indicates the agent is about to call a tool
	EventToolCall EventType = "tool_call"

	// EventToolResult indicates a tool execution has completed
	EventToolResult EventType = "tool_result"

	// EventResponse indicates the agent is streaming a response to the user
	EventResponse EventType = "response"

	// EventError indicates an error occurred during execution
	EventError EventType = "error"

	// EventComplete indicates the agent has finished processing
	EventComplete EventType = "complete"
)

// Event represents something that happened during agent execution.
//
// Events are streamed via channels to provide real-time feedback
// about agent thinking, tool usage, and responses.
type Event struct {
	Type      EventType   // What kind of event this is
	Data      interface{} // Type-specific data (cast based on Type)
	Timestamp time.Time   // When the event occurred
}

// ThinkingEvent contains the agent's internal reasoning.
//
// This is only emitted if ThinkingVisible is true in AgentConfig.
type ThinkingEvent struct {
	Content string // The agent's reasoning/planning thoughts
}

// ToolCallEvent indicates a tool is about to be invoked.
type ToolCallEvent struct {
	ToolName string      // Name of the tool being called
	Input    interface{} // Input parameters for the tool
}

// ToolResultEvent contains the result of a tool execution.
type ToolResultEvent struct {
	ToolName string      // Name of the tool that was called
	Output   interface{} // Output from the tool (nil if error occurred)
	Error    error       // Error if tool execution failed
}

// ResponseEvent contains text being streamed to the user.
type ResponseEvent struct {
	Content string // Text content (may be partial if streaming)
	Done    bool   // True if this is the final chunk of the response
}

// ErrorEvent contains error information.
type ErrorEvent struct {
	Error   error  // The error that occurred
	Message string // Human-readable error message
}
