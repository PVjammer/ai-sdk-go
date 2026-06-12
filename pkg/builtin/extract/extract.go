// Package extract provides data extraction from text using either LLM-based
// arbitrary extraction or regex-based extraction for well-known patterns.
//
// LLM path (--type flag): handles any natural-language type description.
//
//	cat schema.sql   | extract --type "table names"
//	cat access.log   | extract --type "IP addresses" --format json
//	cat main.go      | extract --type "function signatures"
//
// Regex path (--data-types flag): fast, deterministic, no LLM required.
//
//	cat contacts.txt | extract --data-types emails,phones
package extract

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"regexp"
	"strings"

	"github.com/pvjammer/ai-sdk-go/internal/schema"
	"github.com/pvjammer/ai-sdk-go/internal/validation"
	"github.com/pvjammer/ai-sdk-go/pkg/function"
	"github.com/pvjammer/ai-sdk-go/pkg/llm"
)

// ExtractFunction performs data extraction from text.
type ExtractFunction struct {
	function.Base
	provider llm.Provider
}

// Input defines the input schema for extraction.
type Input struct {
	Text string `json:"text" validate:"required,min=1" description:"Text to extract data from"`

	// LLM-based extraction — use for any natural-language type description.
	Type   string `json:"type" description:"What to extract, in natural language: 'table names', 'function signatures', 'error messages', 'IP addresses', etc. When set, uses LLM extraction."`
	Format string `json:"format" description:"Output format for LLM extraction: list (one per line), json (array), csv" default:"list"`

	// Regex-based extraction — fast, for well-known patterns.
	DataTypes  []string `json:"data_types" description:"Regex-based extraction types: emails, phones, urls, dates, times, names, all"`
	Unique     bool     `json:"unique" description:"Return only unique values" default:"true"`
	CaseIgnore bool     `json:"case_ignore" description:"Case-insensitive matching" default:"true"`
}

// Output defines the output structure.
// Result is always populated and is used by the shell's display helper.
type Output struct {
	Result string `json:"result,omitempty"`

	// Populated only for regex-based extraction.
	Emails []string `json:"emails,omitempty"`
	Phones []string `json:"phones,omitempty"`
	URLs   []string `json:"urls,omitempty"`
	Dates  []string `json:"dates,omitempty"`
	Times  []string `json:"times,omitempty"`
	Names  []string `json:"names,omitempty"`
}

var (
	emailRegex = regexp.MustCompile(`[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}`)
	phoneRegex = regexp.MustCompile(`(\+\d{1,3}[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}`)
	urlRegex   = regexp.MustCompile(`https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}[/a-zA-Z0-9._~:/?#\[@!$&'()*+,;=-]*`)
	dateRegex  = regexp.MustCompile(`\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|[A-Z][a-z]+\s+\d{1,2},?\s+\d{4})\b`)
	timeRegex  = regexp.MustCompile(`\b(\d{1,2}:\d{2}(:\d{2})?(\s*[AaPp][Mm])?)\b`)
	nameRegex  = regexp.MustCompile(`\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b`)
)

// New creates a new ExtractFunction with lazy LLM provider resolution.
func New(cfg function.Config) function.AIFunction {
	return &ExtractFunction{Base: function.NewBase(cfg)}
}

func (f *ExtractFunction) resolveProvider() llm.Provider {
	if f.provider != nil {
		return f.provider
	}
	cfg := f.Config()

	providerType := cfg.GetString("extract.llm_provider")
	if providerType == "" {
		providerType = cfg.GetString("llm.provider")
	}

	switch providerType {
	case "ollama", "":
		endpoint := cfg.GetString("extract.llm_endpoint")
		if endpoint == "" {
			endpoint = cfg.GetString("llm.ollama.endpoint")
		}
		if endpoint == "" {
			endpoint = "http://localhost:11434"
		}
		model := cfg.GetString("extract.llm_model")
		if model == "" {
			model = cfg.GetString("llm.ollama.model")
		}
		if model == "" {
			model = "llama3.2:latest"
		}
		p, err := llm.NewOllamaProvider(endpoint, model)
		if err != nil {
			p, _ = llm.NewOllamaProvider("http://localhost:11434", "llama3.2:latest")
		}
		f.provider = p
	default:
		f.provider, _ = llm.NewOllamaProvider("http://localhost:11434", "llama3.2:latest")
	}
	return f.provider
}

// Execute implements the AIFunction interface.
func (f *ExtractFunction) Execute(ctx context.Context, input interface{}) (interface{}, error) {
	var in *Input
	switch v := input.(type) {
	case *Input:
		in = v
	case map[string]interface{}:
		in = f.parseMapInput(v)
	default:
		return nil, function.NewExecutionError(f, "input validation", "invalid input type")
	}

	if err := validation.Validate(in); err != nil {
		return nil, function.WrapValidationError(f, err)
	}
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}

	if strings.TrimSpace(in.Type) != "" {
		return f.extractWithLLM(ctx, in)
	}
	return f.extractWithRegex(in)
}

// extractWithLLM calls the LLM to extract arbitrary types from text.
func (f *ExtractFunction) extractWithLLM(ctx context.Context, in *Input) (*Output, error) {
	format := strings.ToLower(strings.TrimSpace(in.Format))
	if format == "" {
		format = "list"
	}

	var systemPrompt, userPrompt string
	switch format {
	case "json":
		systemPrompt = "You are a precise data extraction assistant. Output only a JSON array of strings. No explanation, no markdown, no code fences — just the raw JSON array."
		userPrompt = fmt.Sprintf(
			"Extract all %s from the following text.\nReturn a JSON array of strings. If nothing is found, return [].\n\nText:\n%s",
			in.Type, in.Text,
		)
	case "csv":
		systemPrompt = "You are a precise data extraction assistant. Output only comma-separated values on a single line. No explanation, no headers."
		userPrompt = fmt.Sprintf(
			"Extract all %s from the following text.\nReturn as comma-separated values. If nothing is found, return an empty string.\n\nText:\n%s",
			in.Type, in.Text,
		)
	default: // list
		systemPrompt = "You are a precise data extraction assistant. Output only the extracted items, one per line. No explanation, no numbering, no labels."
		userPrompt = fmt.Sprintf(
			"Extract all %s from the following text.\nReturn each item on its own line. If nothing is found, output nothing.\n\nText:\n%s",
			in.Type, in.Text,
		)
	}

	opts := llm.DefaultOptions().
		WithMaxTokens(1024).
		WithTemperature(0.1).
		WithSystemPrompt(systemPrompt)

	ch := f.resolveProvider().Ainvoke(ctx, userPrompt, opts)
	resp, err := llm.CollectResponse(ctx, ch)
	if err != nil {
		return nil, function.WrapExecutionError(f, "llm_extraction", err)
	}

	result := strings.TrimSpace(resp.Text)
	if format == "json" {
		result = cleanJSONArray(result)
	}
	return &Output{Result: result}, nil
}

// cleanJSONArray strips markdown fences and validates JSON array output.
func cleanJSONArray(s string) string {
	s = strings.TrimSpace(s)
	if idx := strings.Index(s, "```"); idx >= 0 {
		s = s[idx:]
		s = strings.TrimPrefix(s, "```json")
		s = strings.TrimPrefix(s, "```")
		if end := strings.LastIndex(s, "```"); end >= 0 {
			s = s[:end]
		}
		s = strings.TrimSpace(s)
	}
	var check []interface{}
	if err := json.Unmarshal([]byte(s), &check); err != nil {
		return s
	}
	b, err := json.Marshal(check)
	if err != nil {
		return s
	}
	return string(b)
}

// extractWithRegex runs regex-based extraction for well-known patterns.
func (f *ExtractFunction) extractWithRegex(in *Input) (*Output, error) {
	out := &Output{}

	extractAll := len(in.DataTypes) == 0
	for _, dt := range in.DataTypes {
		if strings.ToLower(dt) == "all" {
			extractAll = true
			break
		}
	}

	if extractAll || containsLower(in.DataTypes, "emails") {
		out.Emails = extractPattern(in.Text, emailRegex, in.Unique)
	}
	if extractAll || containsLower(in.DataTypes, "phones") {
		out.Phones = extractPattern(in.Text, phoneRegex, in.Unique)
	}
	if extractAll || containsLower(in.DataTypes, "urls") {
		out.URLs = extractPattern(in.Text, urlRegex, in.Unique)
	}
	if extractAll || containsLower(in.DataTypes, "dates") {
		out.Dates = extractPattern(in.Text, dateRegex, in.Unique)
	}
	if extractAll || containsLower(in.DataTypes, "times") {
		out.Times = extractPattern(in.Text, timeRegex, in.Unique)
	}
	if extractAll || containsLower(in.DataTypes, "names") {
		out.Names = extractPattern(in.Text, nameRegex, in.Unique)
	}

	out.Result = formatRegexOutput(out)
	return out, nil
}

func extractPattern(text string, re *regexp.Regexp, unique bool) []string {
	matches := re.FindAllString(text, -1)
	if !unique {
		return matches
	}
	seen := make(map[string]bool, len(matches))
	result := make([]string, 0, len(matches))
	for _, m := range matches {
		if !seen[m] {
			seen[m] = true
			result = append(result, m)
		}
	}
	return result
}

func formatRegexOutput(o *Output) string {
	var parts []string
	if len(o.Emails) > 0 {
		parts = append(parts, "Emails:\n  "+strings.Join(o.Emails, "\n  "))
	}
	if len(o.Phones) > 0 {
		parts = append(parts, "Phones:\n  "+strings.Join(o.Phones, "\n  "))
	}
	if len(o.URLs) > 0 {
		parts = append(parts, "URLs:\n  "+strings.Join(o.URLs, "\n  "))
	}
	if len(o.Dates) > 0 {
		parts = append(parts, "Dates:\n  "+strings.Join(o.Dates, "\n  "))
	}
	if len(o.Times) > 0 {
		parts = append(parts, "Times:\n  "+strings.Join(o.Times, "\n  "))
	}
	if len(o.Names) > 0 {
		parts = append(parts, "Names:\n  "+strings.Join(o.Names, "\n  "))
	}
	if len(parts) == 0 {
		return "No data extracted"
	}
	return strings.Join(parts, "\n")
}

func (f *ExtractFunction) parseMapInput(m map[string]interface{}) *Input {
	changed, _ := m["_changed_flags"].(map[string]bool)
	isSet := func(key string) bool { return changed == nil || changed[key] }

	in := &Input{Format: "list", Unique: true, CaseIgnore: true}

	if text, ok := m["text"].(string); ok && isSet("text") {
		in.Text = text
	}
	if t, ok := m["type"].(string); ok && isSet("type") {
		in.Type = t
	}
	if f, ok := m["format"].(string); ok && isSet("format") {
		in.Format = f
	}
	if unique, ok := m["unique"].(bool); ok && isSet("unique") {
		in.Unique = unique
	}
	if ci, ok := m["case_ignore"].(bool); ok && isSet("case_ignore") {
		in.CaseIgnore = ci
	}
	if dts, ok := m["data_types"].([]interface{}); ok {
		for _, dt := range dts {
			if s, ok := dt.(string); ok {
				in.DataTypes = append(in.DataTypes, s)
			}
		}
	} else if dts, ok := m["data_types"].([]string); ok {
		in.DataTypes = dts
	}
	return in
}

// InputSchema implements the AIFunction interface.
func (f *ExtractFunction) InputSchema() *function.Schema {
	return schema.MustGenerateSchema(reflect.TypeOf(Input{}))
}

// Metadata implements the AIFunction interface.
func (f *ExtractFunction) Metadata() *function.Metadata {
	return function.NewMetadata("ExtractFunction").
		WithName("extract").
		WithDescription("Extract structured data from text — arbitrary types via LLM (--type) or well-known patterns via regex (--data-types)").
		WithToolName("extract_data").
		WithToolDescription("Extracts data from text. Use type for LLM-based arbitrary extraction (table names, function signatures, error messages, etc.) or data_types for fast regex extraction of emails/phones/URLs/dates/times/names").
		WithCLIName("extract").
		WithCLIDescription("Extract data from text using AI (--type) or regex (--data-types)").
		WithCategory("data").
		WithTags("data", "extraction", "llm", "regex", "parsing")
}

func containsLower(slice []string, s string) bool {
	for _, v := range slice {
		if strings.ToLower(v) == s {
			return true
		}
	}
	return false
}
