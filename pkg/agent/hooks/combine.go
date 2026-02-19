package hooks

import (
	"context"

	"github.com/pvjammer/ai-sdk-go/pkg/agent"
	"github.com/pvjammer/ai-sdk-go/pkg/llm"
)

// Combine merges multiple AgentHooks into one.
//
// For each hook point, all non-nil implementations run in the order given.
// For transforming hooks (PreInput, PreModel, PostModel, PreTool, PostTool),
// the output of each hook is fed as input to the next — they form a pipeline.
//
// Special cases:
//   - PreTool: if any hook returns proceed=false or a non-nil error, the chain
//     stops immediately and execution is skipped.
//   - PostTool: once a hook processes the result, execErr is passed as nil to
//     subsequent hooks (the first hook to handle the error "owns" it).
//   - PostResponse: all hooks run regardless of individual errors; the last
//     non-nil error is returned.
//
// Example — layer LargeResult under Logging:
//
//	agent.SetHooks(hooks.Combine(
//	    hooks.Logging(os.Stderr),
//	    hooks.LargeResult(8192, ""),
//	))
func Combine(hs ...agent.AgentHooks) agent.AgentHooks {
	return agent.AgentHooks{
		PreInput:     combinePreInput(hs),
		PreModel:     combinePreModel(hs),
		PostModel:    combinePostModel(hs),
		PreTool:      combinePreTool(hs),
		PostTool:     combinePostTool(hs),
		PostResponse: combinePostResponse(hs),
	}
}

func combinePreInput(hs []agent.AgentHooks) func(context.Context, string, string) (string, error) {
	return func(ctx context.Context, sessionID, message string) (string, error) {
		for _, h := range hs {
			if h.PreInput == nil {
				continue
			}
			var err error
			message, err = h.PreInput(ctx, sessionID, message)
			if err != nil {
				return message, err
			}
		}
		return message, nil
	}
}

func combinePreModel(hs []agent.AgentHooks) func(context.Context, []llm.ChatMessage) ([]llm.ChatMessage, error) {
	return func(ctx context.Context, messages []llm.ChatMessage) ([]llm.ChatMessage, error) {
		for _, h := range hs {
			if h.PreModel == nil {
				continue
			}
			var err error
			messages, err = h.PreModel(ctx, messages)
			if err != nil {
				return messages, err
			}
		}
		return messages, nil
	}
}

func combinePostModel(hs []agent.AgentHooks) func(context.Context, llm.StreamChunk) (llm.StreamChunk, error) {
	return func(ctx context.Context, chunk llm.StreamChunk) (llm.StreamChunk, error) {
		for _, h := range hs {
			if h.PostModel == nil {
				continue
			}
			var err error
			chunk, err = h.PostModel(ctx, chunk)
			if err != nil {
				return chunk, err
			}
		}
		return chunk, nil
	}
}

func combinePreTool(hs []agent.AgentHooks) func(context.Context, llm.ToolCall) (llm.ToolCall, bool, error) {
	return func(ctx context.Context, call llm.ToolCall) (llm.ToolCall, bool, error) {
		for _, h := range hs {
			if h.PreTool == nil {
				continue
			}
			var (
				proceed bool
				err     error
			)
			call, proceed, err = h.PreTool(ctx, call)
			if err != nil || !proceed {
				// Short-circuit: denial or error stops the chain
				return call, proceed, err
			}
		}
		return call, true, nil
	}
}

func combinePostTool(hs []agent.AgentHooks) func(context.Context, string, string, error) (string, error) {
	return func(ctx context.Context, toolName, result string, execErr error) (string, error) {
		for _, h := range hs {
			if h.PostTool == nil {
				continue
			}
			var err error
			result, err = h.PostTool(ctx, toolName, result, execErr)
			if err != nil {
				return result, err
			}
			// The first hook that runs "owns" execErr; subsequent hooks see nil
			// so they treat the result as a normal (already-processed) string.
			execErr = nil
		}
		return result, nil
	}
}

func combinePostResponse(hs []agent.AgentHooks) func(context.Context, string) error {
	return func(ctx context.Context, response string) error {
		var lastErr error
		for _, h := range hs {
			if h.PostResponse == nil {
				continue
			}
			if err := h.PostResponse(ctx, response); err != nil {
				lastErr = err
			}
		}
		return lastErr
	}
}
