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

// ReActPlanner implements the Planner interface using the ReAct pattern.
// It builds a prompt that instructs the model to respond with a JSON object
// containing Reasoning, Action, and optional ToolCalls.
// This pattern improves tool call reliability and makes parsing deterministic.
//
// Like SimplePlanner, it uses ConversationProvider for multi-message
// conversations when available, falling back to single-prompt Ainvoke.
type ReActPlanner struct {
	llm          llm.Provider
	systemPrompt string
}

// NewReActPlanner creates a new ReActPlanner.
func NewReActPlanner(provider llm.Provider, systemPrompt string) *ReActPlanner {
	return &ReActPlanner{
		llm:          provider,
		systemPrompt: systemPrompt,
	}
}

// Plan constructs the messages, calls the LLM and parses the JSON response.
func (p *ReActPlanner) Plan(ctx context.Context, messages []agentcontext.Message, tools []ToolDefinition) (*Plan, error) {
	chatMessages := p.buildMessages(messages, tools)
	resp, err := p.callLLM(ctx, chatMessages)
	if err != nil {
		return nil, fmt.Errorf("LLM call failed: %w", err)
	}
	return p.parseResponse(resp)
}

// buildReActToolsText formats tool definitions with ReAct JSON format instructions.
func buildReActToolsText(tools []ToolDefinition) string {
	if len(tools) == 0 {
		return ""
	}

	var sb strings.Builder
	sb.WriteString("\n\n## Available Tools\n\n")

	for _, t := range tools {
		sb.WriteString(fmt.Sprintf("**%s**: %s\n", t.Name, t.Description))
		if len(t.Parameters) > 0 {
			sb.WriteString("  Parameters:\n")
			if props, ok := t.Parameters["properties"].(map[string]interface{}); ok {
				for name, detail := range props {
					if d, ok := detail.(map[string]interface{}); ok {
						if desc, ok := d["description"].(string); ok {
							sb.WriteString(fmt.Sprintf("  - %s: %s\n", name, desc))
						}
					}
				}
			}
		}
		sb.WriteString("\n")
	}

	sb.WriteString("\n## ReAct Format\n\nThe model must respond with JSON: {\n  \"reasoning\": \"...\",\n  \"action\": \"use_tool\" or \"respond\",\n  \"tool_calls\": [ {\"name\": \"toolName\", \"input\": {…} } ]\n}\n")
	return sb.String()
}

// buildMessages constructs a multi-message conversation with proper role attribution.
func (p *ReActPlanner) buildMessages(messages []agentcontext.Message, tools []ToolDefinition) []llm.ChatMessage {
	var chatMessages []llm.ChatMessage

	// System message: system prompt + tool definitions with ReAct format
	systemContent := p.systemPrompt + buildReActToolsText(tools)
	chatMessages = append(chatMessages, llm.ChatMessage{
		Role:    "system",
		Content: systemContent,
	})

	// Conversation history with proper roles
	for _, msg := range messages {
		chatMessages = append(chatMessages, llm.ChatMessage{
			Role:    msg.Role,
			Content: msg.Content,
		})

		// Add tool results as separate "tool" role messages
		for _, tc := range msg.ToolCalls {
			if tc.Output != nil {
				out, _ := json.Marshal(tc.Output)
				chatMessages = append(chatMessages, llm.ChatMessage{
					Role:    "tool",
					Content: fmt.Sprintf("[Tool %s returned: %s]", tc.Name, string(out)),
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

// callLLM invokes the LLM provider using multi-message conversation when available.
func (p *ReActPlanner) callLLM(ctx context.Context, messages []llm.ChatMessage) (string, error) {
	if os.Getenv("AGENT_DEBUG") == "1" {
		fmt.Fprintf(os.Stderr, "\n=== REACT PLANNER DEBUG: LLM MESSAGES (%d) ===\n", len(messages))
		for i, msg := range messages {
			content := msg.Content
			if len(content) > 200 {
				content = content[:200] + "..."
			}
			fmt.Fprintf(os.Stderr, "  [%d] role=%s content=%s\n", i, msg.Role, content)
		}
		fmt.Fprintf(os.Stderr, "=== END MESSAGES ===\n\n")
	}

	opts := &llm.CompletionOptions{MaxTokens: 2048, Temperature: 0.7, Stream: false}

	var chunks <-chan llm.StreamChunk
	if cp, ok := p.llm.(llm.ConversationProvider); ok {
		chunks = cp.ChatMessages(ctx, messages, opts)
	} else {
		prompt := buildFallbackPrompt(messages)
		chunks = p.llm.Ainvoke(ctx, prompt, opts)
	}

	var resp strings.Builder
	for c := range chunks {
		if c.Error != nil {
			return "", c.Error
		}
		resp.WriteString(c.Text)
	}
	return resp.String(), nil
}

// ReActResponse mirrors the expected JSON structure.
type ReActResponse struct {
	Reasoning string                 `json:"reasoning"`
	Action    string                 `json:"action"`
	Response  string                 `json:"response"`
	ToolCalls []agentcontext.ToolCall `json:"tool_calls"`
}

func (p *ReActPlanner) parseResponse(s string) (*Plan, error) {
	var r ReActResponse
	if err := json.Unmarshal([]byte(s), &r); err != nil {
		return nil, fmt.Errorf("failed to parse ReAct JSON: %w", err)
	}
	// Map to planner.Plan
	plan := &Plan{Reasoning: r.Reasoning}
	switch r.Action {
	case string(ActionUseTool):
		plan.Action = ActionUseTool
		plan.ToolCalls = r.ToolCalls
	case string(ActionRespond):
		plan.Action = ActionRespond
		plan.Response = r.Response
	default:
		plan.Action = ActionRespond
		plan.Response = r.Response
	}
	return plan, nil
}
