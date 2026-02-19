package agent

import (
	"context"

	"github.com/pvjammer/ai-sdk-go/pkg/llm"
)

// AgentHooks contains optional lifecycle callbacks injected into the agent loop.
//
// All fields are optional — a nil hook is silently skipped.
// Hooks execute synchronously in the agent goroutine; keep them fast or
// dispatch work to goroutines internally if latency matters.
//
// Execution order per agent loop iteration:
//
//  1. PreInput   — once, when the user message arrives
//  2. PreModel   — before each LLM call (may fire multiple times per Run)
//  3. PostModel  — after each LLM response, before tool dispatch
//  4. PreTool    — before each tool execution (after EventToolCall is emitted)
//  5. PostTool   — after each tool execution, before result enters history
//  6. PostResponse — once, when the agent produces its final text answer
type AgentHooks struct {
	// PreInput transforms the user message before it is added to session history.
	// Return the (possibly modified) message string.
	// Return an error to abort the Run with EventError.
	PreInput func(ctx context.Context, sessionID, message string) (string, error)

	// PreModel is called with the full message slice before each LLM invocation.
	// Can prune, compress, summarize, or annotate messages.
	// Return the (possibly modified) message slice.
	// Return an error to abort the current iteration with EventError.
	PreModel func(ctx context.Context, messages []llm.ChatMessage) ([]llm.ChatMessage, error)

	// PostModel is called with the raw LLM response before tool dispatch.
	// Can modify the chunk (e.g., filter tool calls, inject extra reasoning).
	// Return an error to abort the current iteration with EventError.
	PostModel func(ctx context.Context, chunk llm.StreamChunk) (llm.StreamChunk, error)

	// PreTool is called before each tool execution.
	// It receives the tool call after EventToolCall has been emitted, so the
	// observer already knows what the model requested.
	//
	// Returns:
	//   - modified call (can change Name or Args)
	//   - proceed bool: false skips execution and records a denial in history
	//   - err: non-nil skips execution and records an error result in history
	PreTool func(ctx context.Context, call llm.ToolCall) (llm.ToolCall, bool, error)

	// PostTool is called after each tool execution, before the result string
	// is stored in session history. This is the primary hook for large-result
	// handling — check len(result) and redirect to a file if needed.
	//
	//   toolName  name of the tool that ran
	//   result    JSON-marshaled output string (or "error: ..." if execErr != nil)
	//   execErr   the error from tool execution (nil on success)
	//
	// Return the modified result string to store in history.
	// EventToolResult has already been emitted with the raw output, so the
	// caller's UI always sees the real result; only the model sees the modified
	// history entry.
	PostTool func(ctx context.Context, toolName, result string, execErr error) (string, error)

	// PostResponse is called once with the agent's final text response.
	// It fires after the response has been added to session history and after
	// EventResponse has been emitted, so errors here do not suppress the response.
	// A non-nil error is emitted as EventError but does not abort EventComplete.
	PostResponse func(ctx context.Context, response string) error
}
