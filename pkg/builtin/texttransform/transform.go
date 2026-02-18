// Package texttransform provides text transformation operations.
//
// This package implements a simple text transformation function that demonstrates
// all features of the ai-functions framework: schema generation, validation,
// CLI generation, and LLM tool format.
package texttransform

import (
	"context"
	"reflect"
	"strings"
	"unicode"

	"github.com/pvjammer/ai-sdk-go/internal/schema"
	"github.com/pvjammer/ai-sdk-go/internal/validation"
	"github.com/pvjammer/ai-sdk-go/pkg/function"
)

// TextTransformFunction performs various text transformations.
type TextTransformFunction struct {
	function.Base
}

// Input defines the input schema for text transformation.
type Input struct {
	Text      string `json:"text" validate:"required,min=1" description:"Text to transform"`
	Operation string `json:"operation" validate:"oneof=upper lower title reverse capitalize" enum:"upper,lower,title,reverse,capitalize" description:"Transformation operation to perform" default:"upper"`
	TrimSpace bool   `json:"trim_space" description:"Trim leading/trailing whitespace before processing" default:"true"`
}

// NewDefault creates a new TextTransformFunction with default config.
func NewDefault() *TextTransformFunction {
	return &TextTransformFunction{
		Base: function.NewBase(function.NilConfig{}),
	}
}

// Execute implements the AIFunction interface.
func (f *TextTransformFunction) Execute(ctx context.Context, input interface{}) (interface{}, error) {
	// Type assert input
	var in *Input

	// Handle both *Input and map[string]interface{} (from CLI)
	switch v := input.(type) {
	case *Input:
		in = v
	case map[string]interface{}:
		// Convert map to struct for CLI invocation
		in = &Input{}
		if text, ok := v["text"].(string); ok {
			in.Text = text
		}
		if op, ok := v["operation"].(string); ok {
			in.Operation = op
		}
		if trim, ok := v["trim_space"].(bool); ok {
			in.TrimSpace = trim
		}
		// Set defaults if not provided
		if in.Operation == "" {
			in.Operation = "upper"
		}
		if _, ok := v["trim_space"]; !ok {
			in.TrimSpace = true
		}
	default:
		return nil, function.NewExecutionError(f, "input validation",
			"input must be *Input or map[string]interface{}")
	}

	// Validate input
	if err := validation.Validate(in); err != nil {
		return nil, function.WrapValidationError(f, err)
	}

	// Check context before processing
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	// Trim whitespace if requested
	text := in.Text
	if in.TrimSpace {
		text = strings.TrimSpace(text)
	}

	// Perform transformation
	var result string
	switch in.Operation {
	case "upper":
		result = strings.ToUpper(text)
	case "lower":
		result = strings.ToLower(text)
	case "title":
		result = strings.Title(text)
	case "reverse":
		result = reverseString(text)
	case "capitalize":
		result = capitalizeFirst(text)
	default:
		return nil, function.NewExecutionError(f, "transformation",
			"unknown operation: "+in.Operation)
	}

	return result, nil
}

// InputSchema implements the AIFunction interface.
func (f *TextTransformFunction) InputSchema() *function.Schema {
	return schema.MustGenerateSchema(reflect.TypeOf(Input{}))
}

// Metadata implements the AIFunction interface.
func (f *TextTransformFunction) Metadata() *function.Metadata {
	return function.NewMetadata("TextTransformFunction").
		WithName("text_transform").
		WithDescription("Transform text using various operations").
		WithToolName("text_transform").
		WithToolDescription("Transforms text by applying operations like uppercase, lowercase, title case, reverse, or capitalize").
		WithCLIName("transform").
		WithCLIDescription("Transform text using various operations (upper, lower, title, reverse, capitalize)").
		WithCategory("text").
		WithTags("text", "transform", "string", "formatting").
		WithExamples(
			function.Example{
				Input: map[string]interface{}{
					"text":      "hello world",
					"operation": "upper",
				},
				Output:      "HELLO WORLD",
				Description: "Convert to uppercase",
			},
			function.Example{
				Input: map[string]interface{}{
					"text":      "HELLO WORLD",
					"operation": "lower",
				},
				Output:      "hello world",
				Description: "Convert to lowercase",
			},
			function.Example{
				Input: map[string]interface{}{
					"text":      "hello world",
					"operation": "title",
				},
				Output:      "Hello World",
				Description: "Convert to title case",
			},
			function.Example{
				Input: map[string]interface{}{
					"text":      "hello",
					"operation": "reverse",
				},
				Output:      "olleh",
				Description: "Reverse text",
			},
			function.Example{
				Input: map[string]interface{}{
					"text":      "hello world",
					"operation": "capitalize",
				},
				Output:      "Hello world",
				Description: "Capitalize first letter",
			},
		)
}

// reverseString reverses a string while preserving UTF-8 encoding.
func reverseString(s string) string {
	runes := []rune(s)
	for i, j := 0, len(runes)-1; i < j; i, j = i+1, j-1 {
		runes[i], runes[j] = runes[j], runes[i]
	}
	return string(runes)
}

// capitalizeFirst capitalizes the first letter while preserving the rest.
func capitalizeFirst(s string) string {
	if s == "" {
		return s
	}

	runes := []rune(s)
	if len(runes) == 0 {
		return s
	}

	runes[0] = unicode.ToUpper(runes[0])
	return string(runes)
}
