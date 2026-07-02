package llm

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestParseTextToolCalls_HermesFormat(t *testing.T) {
	text := `I'll search for that.
<tool_call>
{"name": "web_search", "arguments": {"query": "golang AI agent"}}
</tool_call>`

	calls, remaining := ParseTextToolCalls(text)
	require.Len(t, calls, 1)
	assert.Equal(t, "web_search", calls[0].Name)
	assert.Equal(t, "golang AI agent", calls[0].Args["query"])
	assert.NotContains(t, remaining, "<tool_call>")
}

func TestParseTextToolCalls_HermesParametersVariant(t *testing.T) {
	// Some Qwen3 models emit "parameters" instead of "arguments"
	text := `<tool_call>{"name": "read_file", "parameters": {"path": "/tmp/foo"}}</tool_call>`

	calls, _ := ParseTextToolCalls(text)
	require.Len(t, calls, 1)
	assert.Equal(t, "read_file", calls[0].Name)
	assert.Equal(t, "/tmp/foo", calls[0].Args["path"])
}

func TestParseTextToolCalls_MultipleHermesBlocks(t *testing.T) {
	text := `<tool_call>
{"name": "tool_a", "arguments": {"x": 1}}
</tool_call>
Some text between.
<tool_call>
{"name": "tool_b", "arguments": {"y": "hello"}}
</tool_call>`

	calls, _ := ParseTextToolCalls(text)
	require.Len(t, calls, 2)
	assert.Equal(t, "tool_a", calls[0].Name)
	assert.Equal(t, "tool_b", calls[1].Name)
}

func TestParseTextToolCalls_FunctionTagFormat(t *testing.T) {
	// Old Qwen text-format tool calls
	text := `<function=web_search>
<parameter=query>Google Go AI SDK</parameter>
</function>`

	calls, _ := ParseTextToolCalls(text)
	require.Len(t, calls, 1)
	assert.Equal(t, "web_search", calls[0].Name)
	assert.Equal(t, "Google Go AI SDK", calls[0].Args["query"])
}

func TestParseTextToolCalls_FunctionTagJSONBody(t *testing.T) {
	// Function tag with JSON body instead of parameter tags
	text := `<function=summarize>{"text": "hello world", "length": 100}</function>`

	calls, _ := ParseTextToolCalls(text)
	require.Len(t, calls, 1)
	assert.Equal(t, "summarize", calls[0].Name)
	assert.Equal(t, "hello world", calls[0].Args["text"])
}

func TestParseTextToolCalls_BareJSON(t *testing.T) {
	// Entire response is a bare JSON tool call (DeepSeek / Llama fallback)
	text := `{"name": "web_search", "arguments": {"query": "test"}}`

	calls, _ := ParseTextToolCalls(text)
	require.Len(t, calls, 1)
	assert.Equal(t, "web_search", calls[0].Name)
}

func TestParseTextToolCalls_BareJSONNotTriggeredOnProse(t *testing.T) {
	// Prose containing JSON should NOT trigger bare-JSON parser
	text := `Here is an example: {"key": "value"} — but this is just explanation text.`

	calls, remaining := ParseTextToolCalls(text)
	assert.Empty(t, calls)
	assert.Equal(t, text, remaining)
}

func TestParseTextToolCalls_UseToolFormat(t *testing.T) {
	text := `USE_TOOL: web_search {"query": "open source LLMs"}`

	calls, _ := ParseTextToolCalls(text)
	require.Len(t, calls, 1)
	assert.Equal(t, "web_search", calls[0].Name)
	assert.Equal(t, "open source LLMs", calls[0].Args["query"])
}

func TestParseTextToolCalls_ThinkBlocksStripped(t *testing.T) {
	// Thinking blocks may contain <tool_call> examples that are NOT real calls
	text := `<think>
I should use web_search. For example:
<tool_call>{"name": "example_tool", "arguments": {}}</tool_call>
This is just my reasoning.
</think>
<tool_call>
{"name": "real_tool", "arguments": {"q": "actual query"}}
</tool_call>`

	calls, _ := ParseTextToolCalls(text)
	require.Len(t, calls, 1, "should only find the real call outside the think block")
	assert.Equal(t, "real_tool", calls[0].Name)
}

func TestParseTextToolCalls_NoToolCall(t *testing.T) {
	text := `Here is your answer: the capital of France is Paris.`

	calls, remaining := ParseTextToolCalls(text)
	assert.Empty(t, calls)
	assert.Equal(t, text, remaining)
}

func TestParseTextToolCalls_MalformedJSON(t *testing.T) {
	text := `<tool_call>{"name": "broken", "arguments": {not valid json}}</tool_call>`

	calls, _ := ParseTextToolCalls(text)
	assert.Empty(t, calls, "malformed JSON should not produce a call")
}

func TestToolDefsToHermesJSON(t *testing.T) {
	tools := []ToolDef{
		{
			Name:        "web_search",
			Description: "Search the web",
			Parameters: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"query": map[string]interface{}{
						"type":        "string",
						"description": "Search query",
					},
				},
				"required": []string{"query"},
			},
		},
	}

	json, err := ToolDefsToHermesJSON(tools)
	require.NoError(t, err)
	assert.Contains(t, json, `"name": "web_search"`)
	assert.Contains(t, json, `"description": "Search the web"`)
	assert.Contains(t, json, `"parameters"`)
}

func TestInjectHermesTools_AppendsToSystemMessage(t *testing.T) {
	messages := []ChatMessage{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: "Search for something."},
	}
	tools := []ToolDef{{Name: "web_search", Description: "Search", Parameters: nil}}

	result, err := injectHermesTools(messages, tools)
	require.NoError(t, err)
	require.Equal(t, "system", result[0].Role)
	assert.Contains(t, result[0].Content, "You are a helpful assistant.")
	assert.Contains(t, result[0].Content, "<tools>")
	assert.Contains(t, result[0].Content, "web_search")
	assert.Contains(t, result[0].Content, "<tool_call>")
}

func TestInjectHermesTools_CreatesSystemMessageWhenAbsent(t *testing.T) {
	messages := []ChatMessage{
		{Role: "user", Content: "Search for something."},
	}
	tools := []ToolDef{{Name: "web_search", Description: "Search", Parameters: nil}}

	result, err := injectHermesTools(messages, tools)
	require.NoError(t, err)
	require.Equal(t, "system", result[0].Role)
	assert.Contains(t, result[0].Content, "<tools>")
	assert.Equal(t, "user", result[1].Role)
}

func TestInjectHermesTools_NoToolsPassthrough(t *testing.T) {
	messages := []ChatMessage{{Role: "user", Content: "Hello"}}

	result, err := injectHermesTools(messages, nil)
	require.NoError(t, err)
	assert.Equal(t, messages, result)
}
