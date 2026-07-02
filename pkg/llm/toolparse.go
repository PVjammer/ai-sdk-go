package llm

import (
	"encoding/json"
	"regexp"
	"strings"
)

// ParseTextToolCalls extracts tool calls from a model's text response.
// It recognises multiple text-format conventions used by open-source models
// when native tool calling is unavailable or unreliable.
//
// Formats tried in priority order:
//  1. <tool_call>{"name":"...","arguments":{...}}</tool_call>  — Hermes/Qwen3 standard
//  2. <tool_call>{"name":"...","parameters":{...}}</tool_call> — Qwen3 variation
//  3. <function=name>...<parameter=k>v</parameter>...</function> — old Qwen format
//  4. {"name":"...","arguments":{...}} bare JSON                — DeepSeek/Llama fallback
//  5. USE_TOOL: name {...}                                       — SimplePlanner compat
//
// <think>...</think> blocks are stripped before parsing; thinking-mode models
// (Qwen3, DeepSeek-R1) may include <tool_call> examples inside their reasoning
// chain that are not actual invocations.
//
// Returns the parsed calls and the remaining text (prose, stripped tool blocks, etc.).
// If no calls are found, remaining equals the original text.
func ParseTextToolCalls(text string) (calls []ToolCall, remaining string) {
	// Strip thinking blocks first so embedded examples don't become false positives.
	stripped := stripThinkBlocks(text)

	// Try each parser; the first format that yields at least one valid call wins.
	if calls, rem := parseHermesTagCalls(stripped); len(calls) > 0 {
		return calls, rem
	}
	if calls, rem := parseFunctionTagCalls(stripped); len(calls) > 0 {
		return calls, rem
	}
	if calls, rem := parseBareJSONCalls(stripped); len(calls) > 0 {
		return calls, rem
	}
	if calls, rem := parseUseToolCalls(stripped); len(calls) > 0 {
		return calls, rem
	}

	return nil, text
}

// ToolDefsToHermesJSON serializes tool definitions into the JSON array format
// expected inside <tools>...</tools> tags by Hermes/Qwen3 models.
func ToolDefsToHermesJSON(tools []ToolDef) (string, error) {
	type hermesTool struct {
		Name        string      `json:"name"`
		Description string      `json:"description"`
		Parameters  interface{} `json:"parameters"`
	}

	hermes := make([]hermesTool, len(tools))
	for i, td := range tools {
		params := td.Parameters
		if params == nil {
			params = map[string]interface{}{
				"type":       "object",
				"properties": map[string]interface{}{},
			}
		}
		hermes[i] = hermesTool{
			Name:        td.Name,
			Description: td.Description,
			Parameters:  params,
		}
	}

	b, err := json.MarshalIndent(hermes, "", "  ")
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// ── internal parsers ──────────────────────────────────────────────────────────

var reThink = regexp.MustCompile(`(?s)<think>.*?</think>`)

// stripThinkBlocks removes <think>...</think> sections from text.
func stripThinkBlocks(text string) string {
	return strings.TrimSpace(reThink.ReplaceAllString(text, ""))
}

var reHermesTag = regexp.MustCompile(`<tool_call>\s*([\s\S]*?)\s*</tool_call>`)

// parseHermesTagCalls handles <tool_call>{"name":"...","arguments":{...}}</tool_call>.
// The "parameters" key is also accepted as an alias for "arguments".
func parseHermesTagCalls(text string) ([]ToolCall, string) {
	matches := reHermesTag.FindAllStringSubmatchIndex(text, -1)
	if len(matches) == 0 {
		return nil, text
	}

	var calls []ToolCall
	for _, m := range matches {
		jsonStr := text[m[2]:m[3]]
		if call, ok := parseToolCallJSON(jsonStr); ok {
			calls = append(calls, call)
		}
	}

	if len(calls) == 0 {
		return nil, text
	}

	remaining := strings.TrimSpace(reHermesTag.ReplaceAllString(text, ""))
	return calls, remaining
}

var reFunctionTag = regexp.MustCompile(`<function=(\w+)>([\s\S]*?)</function>`)
var reParamTag = regexp.MustCompile(`<parameter=(\w+)>([\s\S]*?)</parameter>`)

// parseFunctionTagCalls handles <function=name>...<parameter=k>v</parameter>...</function>.
func parseFunctionTagCalls(text string) ([]ToolCall, string) {
	matches := reFunctionTag.FindAllStringSubmatchIndex(text, -1)
	if len(matches) == 0 {
		return nil, text
	}

	var calls []ToolCall
	for _, m := range matches {
		name := text[m[2]:m[3]]
		body := text[m[4]:m[5]]

		args := make(map[string]interface{})

		// Try <parameter=key>value</parameter> tags first.
		for _, pm := range reParamTag.FindAllStringSubmatch(body, -1) {
			args[pm[1]] = strings.TrimSpace(pm[2])
		}

		// If no param tags, try interpreting the body as JSON.
		if len(args) == 0 {
			var jsonArgs map[string]interface{}
			if err := json.Unmarshal([]byte(strings.TrimSpace(body)), &jsonArgs); err == nil {
				args = jsonArgs
			}
		}

		calls = append(calls, ToolCall{Name: name, Args: args})
	}

	remaining := strings.TrimSpace(reFunctionTag.ReplaceAllString(text, ""))
	return calls, remaining
}

// parseBareJSONCalls handles a bare {"name":"...","arguments":{...}} object.
// Only applied when the entire stripped response looks like a single JSON object
// to avoid false positives on responses that merely contain JSON examples.
func parseBareJSONCalls(text string) ([]ToolCall, string) {
	trimmed := strings.TrimSpace(text)
	if !strings.HasPrefix(trimmed, "{") {
		return nil, text
	}
	call, ok := parseToolCallJSON(trimmed)
	if !ok {
		return nil, text
	}
	return []ToolCall{call}, ""
}

var reUseTool = regexp.MustCompile(`USE_TOOL:\s*(\w+)\s*\{`)

// parseUseToolCalls handles USE_TOOL: name {...} (SimplePlanner / legacy format).
func parseUseToolCalls(text string) ([]ToolCall, string) {
	idxMatches := reUseTool.FindAllStringSubmatchIndex(text, -1)
	if len(idxMatches) == 0 {
		return nil, text
	}

	type removal struct{ start, end int }
	var calls []ToolCall
	var removals []removal

	for _, m := range idxMatches {
		name := text[m[2]:m[3]]
		braceStart := m[1] - 1 // the '{' at the end of the USE_TOOL: match
		jsonStr, ok := extractBalancedJSON(text, braceStart)
		if !ok {
			continue
		}
		var args map[string]interface{}
		if err := json.Unmarshal([]byte(jsonStr), &args); err != nil {
			continue
		}
		calls = append(calls, ToolCall{Name: name, Args: args})
		removals = append(removals, removal{m[0], braceStart + len(jsonStr)})
	}

	if len(calls) == 0 {
		return nil, text
	}

	var sb strings.Builder
	prev := 0
	for _, r := range removals {
		sb.WriteString(text[prev:r.start])
		prev = r.end
	}
	sb.WriteString(text[prev:])
	return calls, strings.TrimSpace(sb.String())
}

// parseToolCallJSON parses a JSON object and extracts "name" and "arguments"
// (or "parameters") fields as a ToolCall.
func parseToolCallJSON(s string) (ToolCall, bool) {
	var m map[string]json.RawMessage
	if err := json.Unmarshal([]byte(s), &m); err != nil {
		return ToolCall{}, false
	}

	var name string
	if raw, ok := m["name"]; ok {
		_ = json.Unmarshal(raw, &name)
	}
	if name == "" {
		return ToolCall{}, false
	}

	var args map[string]interface{}
	for _, key := range []string{"arguments", "parameters"} {
		if raw, ok := m[key]; ok {
			if err := json.Unmarshal(raw, &args); err == nil {
				break
			}
		}
	}
	if args == nil {
		args = make(map[string]interface{})
	}

	return ToolCall{Name: name, Args: args}, true
}

// extractBalancedJSON extracts a complete JSON object from s starting at position start.
// Handles nested braces and quoted strings containing braces.
func extractBalancedJSON(s string, start int) (string, bool) {
	if start >= len(s) || s[start] != '{' {
		return "", false
	}
	depth := 0
	inStr := false
	escaped := false
	for i := start; i < len(s); i++ {
		ch := s[i]
		if escaped {
			escaped = false
			continue
		}
		if ch == '\\' && inStr {
			escaped = true
			continue
		}
		if ch == '"' {
			inStr = !inStr
			continue
		}
		if inStr {
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
