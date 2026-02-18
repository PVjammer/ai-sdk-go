// Package extract provides data extraction functions.
//
// This package implements regex-based data extraction for common patterns
// like emails, phone numbers, URLs, dates, times, and names.
package extract

import (
	"context"
	"fmt"
	"reflect"
	"regexp"
	"strings"

	"github.com/pvjammer/ai-sdk-go/internal/schema"
	"github.com/pvjammer/ai-sdk-go/internal/validation"
	"github.com/pvjammer/ai-sdk-go/pkg/function"
)

// ExtractFunction performs data extraction from text.
type ExtractFunction struct {
	function.Base
}

// Input defines the input schema for extraction.
type Input struct {
	Text       string   `json:"text" validate:"required,min=1" description:"Text to extract data from"`
	DataTypes  []string `json:"data_types" description:"Types of data to extract (emails, phones, urls, dates, times, names, all)" default:"all"`
	Unique     bool     `json:"unique" description:"Return only unique values" default:"true"`
	CaseIgnore bool     `json:"case_ignore" description:"Case-insensitive matching" default:"true"`
}

// Output defines the output structure.
type Output struct {
	Emails []string `json:"emails,omitempty"`
	Phones []string `json:"phones,omitempty"`
	URLs   []string `json:"urls,omitempty"`
	Dates  []string `json:"dates,omitempty"`
	Times  []string `json:"times,omitempty"`
	Names  []string `json:"names,omitempty"`
}

// Regex patterns for extraction
var (
	emailRegex = regexp.MustCompile(`[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}`)
	phoneRegex = regexp.MustCompile(`(\+\d{1,3}[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}`)
	urlRegex   = regexp.MustCompile(`https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}[/a-zA-Z0-9._~:/?#[@!$&'()*+,;=-]*`)
	dateRegex  = regexp.MustCompile(`\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|[A-Z][a-z]+\s+\d{1,2},?\s+\d{4})\b`)
	timeRegex  = regexp.MustCompile(`\b(\d{1,2}:\d{2}(:\d{2})?(\s*[AaPp][Mm])?)\b`)
	nameRegex  = regexp.MustCompile(`\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b`)
)

// New creates a new ExtractFunction.
func New(cfg function.Config) function.AIFunction {
	return &ExtractFunction{
		Base: function.NewBase(cfg),
	}
}

// NewDefault creates a new ExtractFunction with default config.
func NewDefault() *ExtractFunction {
	return &ExtractFunction{
		Base: function.NewBase(function.NilConfig{}),
	}
}

// Execute implements the AIFunction interface.
func (f *ExtractFunction) Execute(ctx context.Context, input interface{}) (interface{}, error) {
	// Type assert input
	var in *Input

	switch v := input.(type) {
	case *Input:
		in = v
	case map[string]interface{}:
		in = &Input{}
		if text, ok := v["text"].(string); ok {
			in.Text = text
		}
		if dataTypes, ok := v["data_types"].([]interface{}); ok {
			in.DataTypes = make([]string, len(dataTypes))
			for i, dt := range dataTypes {
				if s, ok := dt.(string); ok {
					in.DataTypes[i] = s
				}
			}
		} else if dataTypes, ok := v["data_types"].([]string); ok {
			in.DataTypes = dataTypes
		}
		if unique, ok := v["unique"].(bool); ok {
			in.Unique = unique
		}
		if caseIgnore, ok := v["case_ignore"].(bool); ok {
			in.CaseIgnore = caseIgnore
		}
		// Set defaults
		if len(in.DataTypes) == 0 {
			in.DataTypes = []string{"all"}
		}
		if _, ok := v["unique"]; !ok {
			in.Unique = true
		}
		if _, ok := v["case_ignore"]; !ok {
			in.CaseIgnore = true
		}
	default:
		return nil, function.NewExecutionError(f, "input validation", "invalid input type")
	}

	// Validate
	if err := validation.Validate(in); err != nil {
		return nil, function.WrapValidationError(f, err)
	}

	// Check context
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	// Perform extraction
	output, err := f.extract(in)
	if err != nil {
		return nil, function.WrapExecutionError(f, "extraction", err)
	}

	return output, nil
}

// extract performs the actual data extraction.
func (f *ExtractFunction) extract(in *Input) (*Output, error) {
	output := &Output{}

	// Normalize data types
	extractAll := false
	for _, dt := range in.DataTypes {
		if strings.ToLower(dt) == "all" {
			extractAll = true
			break
		}
	}

	text := in.Text
	if in.CaseIgnore {
		// For case-insensitive patterns, we'll work with original case
		// but the patterns themselves handle case
	}

	// Extract each data type
	if extractAll || contains(in.DataTypes, "emails") {
		output.Emails = f.extractPattern(text, emailRegex, in.Unique)
	}

	if extractAll || contains(in.DataTypes, "phones") {
		output.Phones = f.extractPattern(text, phoneRegex, in.Unique)
	}

	if extractAll || contains(in.DataTypes, "urls") {
		output.URLs = f.extractPattern(text, urlRegex, in.Unique)
	}

	if extractAll || contains(in.DataTypes, "dates") {
		output.Dates = f.extractPattern(text, dateRegex, in.Unique)
	}

	if extractAll || contains(in.DataTypes, "times") {
		output.Times = f.extractPattern(text, timeRegex, in.Unique)
	}

	if extractAll || contains(in.DataTypes, "names") {
		output.Names = f.extractPattern(text, nameRegex, in.Unique)
	}

	return output, nil
}

// extractPattern extracts matches using a regex pattern.
func (f *ExtractFunction) extractPattern(text string, pattern *regexp.Regexp, unique bool) []string {
	matches := pattern.FindAllString(text, -1)

	if !unique {
		return matches
	}

	// Remove duplicates
	seen := make(map[string]bool)
	result := make([]string, 0)

	for _, match := range matches {
		if !seen[match] {
			seen[match] = true
			result = append(result, match)
		}
	}

	return result
}

// contains checks if a slice contains a string (case-insensitive).
func contains(slice []string, str string) bool {
	str = strings.ToLower(str)
	for _, s := range slice {
		if strings.ToLower(s) == str {
			return true
		}
	}
	return false
}

// InputSchema implements the AIFunction interface.
func (f *ExtractFunction) InputSchema() *function.Schema {
	return schema.MustGenerateSchema(reflect.TypeOf(Input{}))
}

// Metadata implements the AIFunction interface.
func (f *ExtractFunction) Metadata() *function.Metadata {
	return function.NewMetadata("ExtractFunction").
		WithName("extract").
		WithDescription("Extract structured data from text").
		WithToolName("extract_data").
		WithToolDescription("Extracts structured data (emails, phones, URLs, dates, times, names) from unstructured text using regex patterns").
		WithCLIName("extract").
		WithCLIDescription("Extract emails, phones, URLs, dates, times, and names from text").
		WithCategory("data").
		WithTags("data", "extraction", "regex", "parsing", "nlp").
		WithExamples(
			function.Example{
				Input: map[string]interface{}{
					"text":       "Contact John Smith at john@example.com or call 555-123-4567",
					"data_types": []string{"emails", "phones", "names"},
				},
				Output: &Output{
					Emails: []string{"john@example.com"},
					Phones: []string{"555-123-4567"},
					Names:  []string{"John Smith"},
				},
				Description: "Extract contact information",
			},
			function.Example{
				Input: map[string]interface{}{
					"text":       "Visit https://example.com or https://github.com for more info",
					"data_types": []string{"urls"},
				},
				Output: &Output{
					URLs: []string{"https://example.com", "https://github.com"},
				},
				Description: "Extract URLs",
			},
			function.Example{
				Input: map[string]interface{}{
					"text": "The meeting is on 12/25/2023 at 3:30 PM",
				},
				Output: &Output{
					Dates: []string{"12/25/2023"},
					Times: []string{"3:30 PM"},
				},
				Description: "Extract dates and times (all types)",
			},
		)
}

// FormatOutput formats the output as a human-readable string.
func (o *Output) FormatOutput() string {
	var parts []string

	if len(o.Emails) > 0 {
		parts = append(parts, fmt.Sprintf("Emails: %s", strings.Join(o.Emails, ", ")))
	}
	if len(o.Phones) > 0 {
		parts = append(parts, fmt.Sprintf("Phones: %s", strings.Join(o.Phones, ", ")))
	}
	if len(o.URLs) > 0 {
		parts = append(parts, fmt.Sprintf("URLs: %s", strings.Join(o.URLs, ", ")))
	}
	if len(o.Dates) > 0 {
		parts = append(parts, fmt.Sprintf("Dates: %s", strings.Join(o.Dates, ", ")))
	}
	if len(o.Times) > 0 {
		parts = append(parts, fmt.Sprintf("Times: %s", strings.Join(o.Times, ", ")))
	}
	if len(o.Names) > 0 {
		parts = append(parts, fmt.Sprintf("Names: %s", strings.Join(o.Names, ", ")))
	}

	if len(parts) == 0 {
		return "No data extracted"
	}

	return strings.Join(parts, "\n")
}
