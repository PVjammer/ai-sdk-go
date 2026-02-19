package llm

import "context"

// ToolDef describes a tool available to the LLM via native tool calling.
//
// The Parameters field should be a JSON-serializable value that represents
// a JSON Schema object (e.g., *function.Schema or map[string]interface{}).
type ToolDef struct {
	Name        string
	Description string
	Parameters  interface{} // JSON Schema object
}

// ToolCall represents a single tool invocation requested by the LLM.
type ToolCall struct {
	// ID is the tool call identifier (may be empty depending on the provider).
	ID string

	// Name is the name of the tool to invoke.
	Name string

	// Args contains the arguments for the tool call.
	Args map[string]interface{}
}

// ToolCallingProvider extends ConversationProvider with native tool calling support.
//
// Providers that implement this interface pass tool definitions as structured JSON
// and receive structured tool_calls back — no regex parsing required.
type ToolCallingProvider interface {
	ConversationProvider

	// ChatWithTools sends a conversation with tool definitions to the LLM and
	// returns a channel of StreamChunks.
	//
	// When the model decides to call a tool, the final StreamChunk will have
	// ToolCalls set (and Text may be empty). When the model produces a text
	// response, ToolCalls will be empty and Text will contain the response.
	ChatWithTools(ctx context.Context, messages []ChatMessage, tools []ToolDef, opts *CompletionOptions) <-chan StreamChunk
}
