package experiment

// TurnMetric is the canonical shape of one --metrics-file JSON line, written by
// examples/tui-agent (metrics.go) and read back by report.go. Both sides import this package
// for it rather than each keeping their own copy — report.go used to hand-duplicate this shape
// in an unexported struct, kept in sync only by a comment asking a future editor to remember;
// json.Unmarshal ignores unknown/missing fields instead of erroring, so a field rename on one
// side would have silently desynced from the other with no error anywhere.
type TurnMetric struct {
	Timestamp        string             `json:"timestamp"`
	DurationMS       int64              `json:"duration_ms"`
	Message          string             `json:"message"`
	ToolCalls        []ToolCallMetric   `json:"tool_calls"`
	Compactions      []CompactionMetric `json:"compactions"`
	Error            string             `json:"error,omitempty"`
	ResponseChars    int                `json:"response_chars"`
	PromptTokens     int                `json:"prompt_tokens"`
	CompletionTokens int                `json:"completion_tokens"`
}

type ToolCallMetric struct {
	Name string                 `json:"name"`
	Args map[string]interface{} `json:"args,omitempty"`
}

type CompactionMetric struct {
	TokensBefore int `json:"tokens_before"`
	TokensAfter  int `json:"tokens_after"`
}
