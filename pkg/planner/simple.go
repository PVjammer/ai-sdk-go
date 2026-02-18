package planner

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"regexp"
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

// buildToolsText formats tool definitions as text for inclusion in the system prompt.
func buildToolsText(tools []ToolDefinition) string {
	if len(tools) == 0 {
		return ""
	}

	var sb strings.Builder
	sb.WriteString("\n\n## Available Tools\n\n")
	sb.WriteString("You can use the following tools to help accomplish tasks:\n\n")

	for _, tool := range tools {
		sb.WriteString(fmt.Sprintf("**%s**: %s\n", tool.Name, tool.Description))

		// Format parameters concisely
		if len(tool.Parameters) > 0 {
			sb.WriteString("  Parameters:\n")
			if props, ok := tool.Parameters["properties"].(map[string]interface{}); ok {
				for param, details := range props {
					if detailsMap, ok := details.(map[string]interface{}); ok {
						desc := detailsMap["description"]
						sb.WriteString(fmt.Sprintf("  - %s: %v\n", param, desc))
					}
				}
			}
		}
		sb.WriteString("\n")
	}

	sb.WriteString("## Tool Usage Format\n\n")
	sb.WriteString("To use a tool, respond with:\n")
	sb.WriteString("USE_TOOL: tool_name {\"param1\": \"value1\", \"param2\": \"value2\"}\n\n")
	sb.WriteString("To respond to the user without using a tool, just write your response normally.\n")

	return sb.String()
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

// extractJSON extracts a JSON object from a string starting at the given position.
// It handles nested braces and strings containing braces correctly.
func extractJSON(s string, start int) (string, bool) {
	if start >= len(s) || s[start] != '{' {
		return "", false
	}

	depth := 0
	inString := false
	escaped := false

	for i := start; i < len(s); i++ {
		ch := s[i]

		if escaped {
			escaped = false
			continue
		}

		if ch == '\\' && inString {
			escaped = true
			continue
		}

		if ch == '"' {
			inString = !inString
			continue
		}

		if inString {
			continue
		}

		if ch == '{' {
			depth++
		} else if ch == '}' {
			depth--
			if depth == 0 {
				return s[start : i+1], true
			}
		}
	}

	return "", false
}

// parseResponse parses the LLM's response to determine the action.
//
// Looks for "USE_TOOL: name {...}" pattern, otherwise treats as text response.
func (p *SimplePlanner) parseResponse(response string) (*Plan, error) {
	// Step 1: Find the tool name and the position of the opening brace
	toolPattern := regexp.MustCompile(`USE_TOOL:\s*(\w+)\s*\{`)
	matches := toolPattern.FindStringSubmatchIndex(response)

	if os.Getenv("AGENT_DEBUG") == "1" {
		fmt.Fprintf(os.Stderr, "=== PLANNER DEBUG: PARSE RESPONSE ===\n")
		fmt.Fprintf(os.Stderr, "Pattern: USE_TOOL:\\s*(\\w+)\\s*\\{\n")
		fmt.Fprintf(os.Stderr, "Match indices: %v\n", matches)
		fmt.Fprintf(os.Stderr, "=== END PARSE ===\n\n")
	}

	if len(matches) >= 4 {
		// matches[2]:matches[3] is the tool name capture group
		toolName := response[matches[2]:matches[3]]

		// The opening brace is at the end of the full match minus 1
		braceStart := matches[1] - 1

		// Step 2: Extract the full JSON object using brace counting
		jsonInput, ok := extractJSON(response, braceStart)

		if os.Getenv("AGENT_DEBUG") == "1" {
			fmt.Fprintf(os.Stderr, "=== TOOL CALL DETECTED ===\n")
			fmt.Fprintf(os.Stderr, "Tool name: %s\n", toolName)
			fmt.Fprintf(os.Stderr, "JSON input: %s\n", jsonInput)
			fmt.Fprintf(os.Stderr, "JSON extracted: %v\n", ok)
			fmt.Fprintf(os.Stderr, "=== END TOOL CALL ===\n\n")
		}

		if !ok {
			return nil, fmt.Errorf("failed to extract JSON object from tool call")
		}

		// Parse JSON input
		var input map[string]interface{}
		if err := json.Unmarshal([]byte(jsonInput), &input); err != nil {
			if os.Getenv("AGENT_DEBUG") == "1" {
				fmt.Fprintf(os.Stderr, "ERROR: Failed to parse JSON: %v\n", err)
			}
			return nil, fmt.Errorf("failed to parse tool input JSON: %w", err)
		}

		return &Plan{
			Action: ActionUseTool,
			ToolCalls: []agentcontext.ToolCall{
				{
					Name:  toolName,
					Input: input,
				},
			},
			Reasoning: "Decided to use tool: " + toolName,
		}, nil
	}

	// No tool call, just a text response
	if os.Getenv("AGENT_DEBUG") == "1" {
		fmt.Fprintf(os.Stderr, "=== NO TOOL CALL - DIRECT RESPONSE ===\n\n")
	}

	return &Plan{
		Action:    ActionRespond,
		Response:  response,
		Reasoning: "Responding directly to user",
	}, nil
}
