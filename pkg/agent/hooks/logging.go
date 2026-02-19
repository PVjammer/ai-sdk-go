package hooks

import (
	"context"
	"fmt"
	"io"
	"time"

	"github.com/pvjammer/ai-sdk-go/pkg/agent"
	"github.com/pvjammer/ai-sdk-go/pkg/llm"
)

// Logging returns an AgentHooks that logs each hook point to w.
//
// All values are passed through unchanged — this hook is purely observational
// and is useful for debugging agent behavior during development.
//
// Example output:
//
//	[agent 15:04:05.123] PreInput  session=abc123 msg_len=42
//	[agent 15:04:05.456] PreModel  messages=3
//	[agent 15:04:05.789] PostModel tools=1 text_len=0
//	[agent 15:04:05.791] PreTool   name=read_file
//	[agent 15:04:05.834] PostTool  name=read_file result_len=2048
//	[agent 15:04:06.100] PostResponse len=120
func Logging(w io.Writer) agent.AgentHooks {
	log := func(format string, args ...any) {
		fmt.Fprintf(w, "[agent %s] %s\n",
			time.Now().Format("15:04:05.000"),
			fmt.Sprintf(format, args...),
		)
	}

	return agent.AgentHooks{
		PreInput: func(ctx context.Context, sessionID, message string) (string, error) {
			log("PreInput  session=%s msg_len=%d", sessionID, len(message))
			return message, nil
		},

		PreModel: func(ctx context.Context, messages []llm.ChatMessage) ([]llm.ChatMessage, error) {
			totalChars := 0
			for _, m := range messages {
				totalChars += len(m.Content)
			}
			log("PreModel  messages=%d total_chars=%d", len(messages), totalChars)
			return messages, nil
		},

		PostModel: func(ctx context.Context, chunk llm.StreamChunk) (llm.StreamChunk, error) {
			log("PostModel tools=%d text_len=%d", len(chunk.ToolCalls), len(chunk.Text))
			return chunk, nil
		},

		PreTool: func(ctx context.Context, call llm.ToolCall) (llm.ToolCall, bool, error) {
			log("PreTool   name=%s id=%s", call.Name, call.ID)
			return call, true, nil
		},

		PostTool: func(ctx context.Context, toolName, result string, execErr error) (string, error) {
			if execErr != nil {
				log("PostTool  name=%s error=%v", toolName, execErr)
			} else {
				log("PostTool  name=%s result_len=%d", toolName, len(result))
			}
			return result, nil
		},

		PostResponse: func(ctx context.Context, response string) error {
			log("PostResponse len=%d", len(response))
			return nil
		},
	}
}
