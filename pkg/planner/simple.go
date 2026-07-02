package planner

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/pvjammer/ai-sdk-go/pkg/agentcontext"
	"github.com/pvjammer/ai-sdk-go/pkg/llm"
)

// SimplePlanner uses prompt engineering to enable tool calling.
//
// This works with ANY model (no native function calling required).
// It formats tools in the system prompt and parses "USE_TOOL: name {...}"
// from the model's response.
//
// When the LLM provider implements ConversationProvider, messages are sent
// with proper role attribution (system, user, assistant, tool). Otherwise
// falls back to packing everything into a single prompt via Ainvoke.
type SimplePlanner struct {
	llm          llm.Provider
	systemPrompt string
}

// NewSimplePlanner creates a prompt-based planner.
func NewSimplePlanner(provider llm.Provider, systemPrompt string) *SimplePlanner {
	return &SimplePlanner{
		llm:          provider,
		systemPrompt: systemPrompt,
	}
}

// Plan implements the Planner interface using prompt-based tool calling.
func (p *SimplePlanner) Plan(ctx context.Context, messages []agentcontext.Message, tools []ToolDefinition) (*Plan, error) {
	// Build multi-message conversation
	chatMessages := p.buildMessages(messages, tools)

	// Call LLM
	response, err := p.callLLM(ctx, chatMessages)
	if err != nil {
		return nil, fmt.Errorf("LLM call failed: %w", err)
	}

	// Parse the response to determine action
	return p.parseResponse(response)
}

// buildToolsText formats tool definitions in Hermes/Qwen3 XML format for inclusion
// in the system prompt. This is compatible with all models fine-tuned on the Hermes
// tool-use dataset (Qwen3, Llama-3.1+, Mistral, DeepSeek, etc.).
func buildToolsText(tools []ToolDefinition) string {
	if len(tools) == 0 {
		return ""
	}

	type toolSchema struct {
		Name        string                 `json:"name"`
		Description string                 `json:"description"`
		Parameters  map[string]interface{} `json:"parameters"`
	}

	schemas := make([]toolSchema, len(tools))
	for i, t := range tools {
		params := t.Parameters
		if params == nil {
			params = map[string]interface{}{
				"type":       "object",
				"properties": map[string]interface{}{},
			}
		}
		schemas[i] = toolSchema{Name: t.Name, Description: t.Description, Parameters: params}
	}

	jsonBytes, _ := json.MarshalIndent(schemas, "", "  ")

	return "\n\n<tools>\n" + string(jsonBytes) + "\n</tools>\n\n" +
		"When you need to call a tool, respond with:\n" +
		"<tool_call>\n{\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n</tool_call>\n" +
		"To respond without a tool, write your answer in plain text."
}

// buildMessages constructs a multi-message conversation with proper role attribution.
//
// This is the key fix for the "invalid provider response" bug: instead of packing
// the entire conversation into a single user message, each message retains its
// original role (system, user, assistant, tool).
func (p *SimplePlanner) buildMessages(messages []agentcontext.Message, tools []ToolDefinition) []llm.ChatMessage {
	var chatMessages []llm.ChatMessage

	// System message: system prompt + tool definitions
	systemContent := p.systemPrompt + buildToolsText(tools)
	chatMessages = append(chatMessages, llm.ChatMessage{
		Role:    "system",
		Content: systemContent,
	})

	// Conversation history with proper roles
	for _, msg := range messages {
		// Add the main message with its original role
		chatMessages = append(chatMessages, llm.ChatMessage{
			Role:    msg.Role,
			Content: msg.Content,
		})

		// Add tool results as separate "tool" role messages
		for _, tc := range msg.ToolCalls {
			if tc.Output != nil {
				formatted := formatToolOutput(tc.Name, tc.Output)
				chatMessages = append(chatMessages, llm.ChatMessage{
					Role:    "tool",
					Content: fmt.Sprintf("[Tool %s returned: %s]", tc.Name, formatted),
				})
			} else if tc.Error != nil {
				chatMessages = append(chatMessages, llm.ChatMessage{
					Role:    "tool",
					Content: fmt.Sprintf("[Tool %s failed: %v]", tc.Name, tc.Error),
				})
			}
		}
	}

	return chatMessages
}

// buildFallbackPrompt flattens multi-message conversation into a single prompt string.
// Used when the provider doesn't implement ConversationProvider.
func buildFallbackPrompt(messages []llm.ChatMessage) string {
	var sb strings.Builder
	for _, msg := range messages {
		switch msg.Role {
		case "system":
			sb.WriteString(msg.Content)
			sb.WriteString("\n\n")
		case "tool":
			sb.WriteString(msg.Content)
			sb.WriteString("\n")
		default:
			sb.WriteString(fmt.Sprintf("%s: %s\n", strings.Title(msg.Role), msg.Content))
		}
	}
	return sb.String()
}

// callLLM invokes the LLM provider using multi-message conversation when available.
//
// If the provider implements ConversationProvider, sends messages with proper
// role attribution via ChatMessages(). Otherwise falls back to Ainvoke() with
// a flattened single prompt.
func (p *SimplePlanner) callLLM(ctx context.Context, messages []llm.ChatMessage) (string, error) {
	// Debug logging
	if os.Getenv("AGENT_DEBUG") == "1" {
		fmt.Fprintf(os.Stderr, "\n=== PLANNER DEBUG: LLM MESSAGES (%d) ===\n", len(messages))
		for i, msg := range messages {
			content := msg.Content
			if len(content) > 200 {
				content = content[:200] + "..."
			}
			fmt.Fprintf(os.Stderr, "  [%d] role=%s content=%s\n", i, msg.Role, content)
		}
		fmt.Fprintf(os.Stderr, "=== END MESSAGES ===\n\n")
	}

	opts := &llm.CompletionOptions{
		MaxTokens:   2048,
		Temperature: 0.7,
		Stream:      false,
	}

	// Use ConversationProvider if available (multi-message)
	var chunks <-chan llm.StreamChunk
	if cp, ok := p.llm.(llm.ConversationProvider); ok {
		if os.Getenv("AGENT_DEBUG") == "1" {
			fmt.Fprintf(os.Stderr, "=== USING ConversationProvider.ChatMessages() ===\n\n")
		}
		chunks = cp.ChatMessages(ctx, messages, opts)
	} else {
		// Fallback: flatten to single prompt
		if os.Getenv("AGENT_DEBUG") == "1" {
			fmt.Fprintf(os.Stderr, "=== FALLBACK: Ainvoke() with flattened prompt ===\n\n")
		}
		prompt := buildFallbackPrompt(messages)
		chunks = p.llm.Ainvoke(ctx, prompt, opts)
	}

	// Collect full response
	var response strings.Builder
	for chunk := range chunks {
		if chunk.Error != nil {
			return "", chunk.Error
		}
		response.WriteString(chunk.Text)
	}

	result := response.String()

	// Debug logging
	if os.Getenv("AGENT_DEBUG") == "1" {
		fmt.Fprintf(os.Stderr, "=== PLANNER DEBUG: LLM RESPONSE ===\n%s\n=== END RESPONSE ===\n\n", result)
	}

	return result, nil
}

// markdownFormatter is implemented by tool outputs that can format themselves as markdown.
type markdownFormatter interface {
	FormatMarkdown() string
}

// formatToolOutput formats tool output concisely for LLM consumption.
func formatToolOutput(toolName string, output interface{}) string {
	// Check if the output knows how to format itself as markdown
	if mf, ok := output.(markdownFormatter); ok {
		result := mf.FormatMarkdown()
		if len(result) > 4000 {
			result = result[:4000] + "\n...[truncated]"
		}
		return result
	}

	// For other tools, use compact JSON
	if jsonBytes, err := json.Marshal(output); err == nil {
		result := string(jsonBytes)
		if len(result) > 2000 {
			result = result[:2000] + "...[truncated]"
		}
		return result
	}

	// Fallback
	return fmt.Sprintf("%v", output)
}

// parseResponse parses the LLM's response to determine the action.
// Uses llm.ParseTextToolCalls to support Hermes, old Qwen, USE_TOOL, and bare-JSON formats.
func (p *SimplePlanner) parseResponse(response string) (*Plan, error) {
	if os.Getenv("AGENT_DEBUG") == "1" {
		fmt.Fprintf(os.Stderr, "=== PLANNER DEBUG: PARSE RESPONSE ===\n%s\n=== END ===\n\n", response)
	}

	calls, _ := llm.ParseTextToolCalls(response)

	if len(calls) > 0 {
		agentCalls := make([]agentcontext.ToolCall, len(calls))
		for i, c := range calls {
			agentCalls[i] = agentcontext.ToolCall{
				Name:  c.Name,
				Input: c.Args,
			}
		}

		if os.Getenv("AGENT_DEBUG") == "1" {
			fmt.Fprintf(os.Stderr, "=== TOOL CALLS DETECTED: %d ===\n", len(calls))
			for _, c := range calls {
				fmt.Fprintf(os.Stderr, "  tool=%s args=%v\n", c.Name, c.Args)
			}
			fmt.Fprintf(os.Stderr, "=== END TOOL CALLS ===\n\n")
		}

		return &Plan{
			Action:    ActionUseTool,
			ToolCalls: agentCalls,
			Reasoning: "Decided to use tool: " + calls[0].Name,
		}, nil
	}

	if os.Getenv("AGENT_DEBUG") == "1" {
		fmt.Fprintf(os.Stderr, "=== NO TOOL CALL - DIRECT RESPONSE ===\n\n")
	}

	return &Plan{
		Action:    ActionRespond,
		Response:  response,
		Reasoning: "Responding directly to user",
	}, nil
}
