package llm

import "testing"

// TestOaiMessage_FinalText_ReasoningFallback guards against a regression found by live testing
// against Ollama: a response truncated by the length cap (a reasoning model spending its whole
// output budget on <think>...) comes back from Ollama's OpenAI-compat endpoint as
// {"content":"","reasoning":"...","finish_reason":"length"} — the field is named "reasoning",
// not the DeepSeek-style "reasoning_content" this code originally only checked for. Without the
// "reasoning" fallback, finalText() returned "", which the agent loop reports as a hard
// "LLM returned empty response" error even though the server actually sent real content.
func TestOaiMessage_FinalText_ReasoningFallback(t *testing.T) {
	strp := func(s string) *string { return &s }

	tests := []struct {
		name string
		msg  oaiMessage
		want string
	}{
		{
			name: "normal content, no reasoning involved",
			msg:  oaiMessage{Content: strp("the answer")},
			want: "the answer",
		},
		{
			name: "empty content falls back to Ollama's \"reasoning\" field",
			msg:  oaiMessage{Content: strp(""), Reasoning: strp("thinking about the answer")},
			want: "thinking about the answer",
		},
		{
			name: "empty content falls back to DeepSeek-style reasoning_content",
			msg:  oaiMessage{Content: strp(""), ReasoningContent: strp("thinking about the answer")},
			want: "thinking about the answer",
		},
		{
			name: "both reasoning fields present prefers reasoning_content",
			msg:  oaiMessage{Content: strp(""), ReasoningContent: strp("primary"), Reasoning: strp("secondary")},
			want: "primary",
		},
		{
			name: "content present alongside reasoning uses content, not reasoning",
			msg:  oaiMessage{Content: strp("the real answer"), Reasoning: strp("some thinking")},
			want: "the real answer",
		},
		{
			name: "nil content, nil reasoning fields: genuinely empty, not an error here",
			msg:  oaiMessage{Content: nil},
			want: "",
		},
		{
			name: "content is a dangling unclosed <think> block with no reasoning field: still empty",
			msg:  oaiMessage{Content: strp("<think>never finished reasoning")},
			want: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.msg.finalText(); got != tt.want {
				t.Errorf("finalText() = %q, want %q", got, tt.want)
			}
		})
	}
}
