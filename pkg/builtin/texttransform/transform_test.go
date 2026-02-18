package texttransform

import (
	"context"
	"testing"

	"github.com/pvjammer/ai-sdk-go/pkg/function"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestNew tests function creation
func TestNew(t *testing.T) {
	fn := NewDefault()
	require.NotNil(t, fn)
	assert.NotNil(t, fn.Config())
}

// TestExecute_Upper tests uppercase transformation
func TestExecute_Upper(t *testing.T) {
	fn := NewDefault()
	ctx := context.Background()

	result, err := fn.Execute(ctx, &Input{
		Text:      "hello world",
		Operation: "upper",
	})

	require.NoError(t, err)
	assert.Equal(t, "HELLO WORLD", result)
}

// TestExecute_Lower tests lowercase transformation
func TestExecute_Lower(t *testing.T) {
	fn := NewDefault()
	ctx := context.Background()

	result, err := fn.Execute(ctx, &Input{
		Text:      "HELLO WORLD",
		Operation: "lower",
	})

	require.NoError(t, err)
	assert.Equal(t, "hello world", result)
}

// TestExecute_Title tests title case transformation
func TestExecute_Title(t *testing.T) {
	fn := NewDefault()
	ctx := context.Background()

	result, err := fn.Execute(ctx, &Input{
		Text:      "hello world",
		Operation: "title",
	})

	require.NoError(t, err)
	assert.Equal(t, "Hello World", result)
}

// TestExecute_Reverse tests text reversal
func TestExecute_Reverse(t *testing.T) {
	fn := NewDefault()
	ctx := context.Background()

	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "simple text",
			input:    "hello",
			expected: "olleh",
		},
		{
			name:     "with spaces",
			input:    "hello world",
			expected: "dlrow olleh",
		},
		{
			name:     "single character",
			input:    "a",
			expected: "a",
		},
		{
			name:     "unicode",
			input:    "hello 世界",
			expected: "界世 olleh",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := fn.Execute(ctx, &Input{
				Text:      tt.input,
				Operation: "reverse",
			})

			require.NoError(t, err)
			assert.Equal(t, tt.expected, result)
		})
	}
}

// TestExecute_Capitalize tests capitalize operation
func TestExecute_Capitalize(t *testing.T) {
	fn := NewDefault()
	ctx := context.Background()

	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "lowercase",
			input:    "hello world",
			expected: "Hello world",
		},
		{
			name:     "already capitalized",
			input:    "Hello world",
			expected: "Hello world",
		},
		{
			name:     "all caps",
			input:    "HELLO WORLD",
			expected: "HELLO WORLD",
		},
		{
			name:     "single char",
			input:    "h",
			expected: "H",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := fn.Execute(ctx, &Input{
				Text:      tt.input,
				Operation: "capitalize",
			})

			require.NoError(t, err)
			assert.Equal(t, tt.expected, result)
		})
	}
}

// TestExecute_TrimSpace tests whitespace trimming
func TestExecute_TrimSpace(t *testing.T) {
	fn := NewDefault()
	ctx := context.Background()

	tests := []struct {
		name      string
		text      string
		trimSpace bool
		expected  string
	}{
		{
			name:      "trim enabled (default)",
			text:      "  hello world  ",
			trimSpace: true,
			expected:  "HELLO WORLD",
		},
		{
			name:      "trim disabled",
			text:      "  hello world  ",
			trimSpace: false,
			expected:  "  HELLO WORLD  ",
		},
		{
			name:      "no whitespace",
			text:      "hello",
			trimSpace: true,
			expected:  "HELLO",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := fn.Execute(ctx, &Input{
				Text:      tt.text,
				Operation: "upper",
				TrimSpace: tt.trimSpace,
			})

			require.NoError(t, err)
			assert.Equal(t, tt.expected, result)
		})
	}
}

// TestExecute_MapInput tests map[string]interface{} input (CLI mode)
func TestExecute_MapInput(t *testing.T) {
	fn := NewDefault()
	ctx := context.Background()

	input := map[string]interface{}{
		"text":      "hello world",
		"operation": "upper",
	}

	result, err := fn.Execute(ctx, input)

	require.NoError(t, err)
	assert.Equal(t, "HELLO WORLD", result)
}

// TestExecute_MapInput_Defaults tests default values with map input
func TestExecute_MapInput_Defaults(t *testing.T) {
	fn := NewDefault()
	ctx := context.Background()

	// Only provide text, operation should default to "upper"
	input := map[string]interface{}{
		"text": "hello",
	}

	result, err := fn.Execute(ctx, input)

	require.NoError(t, err)
	assert.Equal(t, "HELLO", result) // Default operation is upper
}

// TestExecute_ValidationError tests validation errors
func TestExecute_ValidationError(t *testing.T) {
	fn := NewDefault()
	ctx := context.Background()

	tests := []struct {
		name        string
		input       *Input
		expectedErr string
	}{
		{
			name: "empty text",
			input: &Input{
				Text:      "",
				Operation: "upper",
			},
			expectedErr: "validation failed",
		},
		{
			name: "invalid operation",
			input: &Input{
				Text:      "hello",
				Operation: "invalid",
			},
			expectedErr: "validation failed",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := fn.Execute(ctx, tt.input)

			require.Error(t, err)
			assert.Contains(t, err.Error(), tt.expectedErr)
		})
	}
}

// TestExecute_InvalidInput tests invalid input type
func TestExecute_InvalidInput(t *testing.T) {
	fn := NewDefault()
	ctx := context.Background()

	_, err := fn.Execute(ctx, "invalid input type")

	require.Error(t, err)
	assert.Contains(t, err.Error(), "input must be")
}

// TestExecute_ContextCancellation tests context cancellation
func TestExecute_ContextCancellation(t *testing.T) {
	fn := NewDefault()
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	_, err := fn.Execute(ctx, &Input{
		Text:      "hello",
		Operation: "upper",
	})

	require.Error(t, err)
	assert.Equal(t, context.Canceled, err)
}

// TestInputSchema tests schema generation
func TestInputSchema(t *testing.T) {
	fn := NewDefault()
	schema := fn.InputSchema()

	require.NotNil(t, schema)
	assert.Equal(t, "object", schema.Type)

	// Check properties
	assert.Contains(t, schema.Properties, "text")
	assert.Contains(t, schema.Properties, "operation")
	assert.Contains(t, schema.Properties, "trim_space")

	// Check text property
	textProp := schema.Properties["text"]
	assert.Equal(t, "string", textProp.Type)
	assert.Equal(t, "Text to transform", textProp.Description)

	// Check operation property
	opProp := schema.Properties["operation"]
	assert.Equal(t, "string", opProp.Type)
	assert.NotNil(t, opProp.Enum)
	assert.Contains(t, opProp.Enum, "upper")
	assert.Contains(t, opProp.Enum, "lower")
	assert.Equal(t, "upper", opProp.Default)

	// Check required fields
	assert.Contains(t, schema.Required, "text")
}

// TestMetadata tests metadata generation
func TestMetadata(t *testing.T) {
	fn := NewDefault()
	meta := fn.Metadata()

	require.NotNil(t, meta)

	// Check names
	assert.Equal(t, "text_transform", meta.GetName("tool"))
	assert.Equal(t, "transform", meta.GetName("cli"))
	assert.Equal(t, "text_transform", meta.GetName(""))

	// Check descriptions
	assert.NotEmpty(t, meta.GetDescription("tool"))
	assert.NotEmpty(t, meta.GetDescription("cli"))

	// Check category and tags
	assert.Equal(t, "text", meta.Category)
	assert.Contains(t, meta.Tags, "text")
	assert.Contains(t, meta.Tags, "transform")

	// Check examples
	assert.Len(t, meta.Examples, 5)

	// Verify example structure
	for _, example := range meta.Examples {
		assert.NotNil(t, example.Input)
		assert.NotNil(t, example.Output)
		assert.NotEmpty(t, example.Description)
	}
}

// TestAsTool tests LLM tool format generation
func TestAsTool(t *testing.T) {
	fn := NewDefault()
	tool := function.AsTool(fn)

	require.NotNil(t, tool)
	assert.Equal(t, "function", tool["type"])

	funcDef := tool["function"].(map[string]interface{})
	assert.Equal(t, "text_transform", funcDef["name"])
	assert.NotEmpty(t, funcDef["description"])

	// Check parameters
	params := funcDef["parameters"].(map[string]interface{})
	assert.Equal(t, "object", params["type"])

	props := params["properties"].(map[string]interface{})
	assert.Contains(t, props, "text")
	assert.Contains(t, props, "operation")

	// Check examples are included
	assert.NotNil(t, funcDef["examples"])
	examples := funcDef["examples"].([]function.Example)
	assert.Len(t, examples, 5)
}

// TestReverseString tests the reverse helper function
func TestReverseString(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"", ""},
		{"a", "a"},
		{"ab", "ba"},
		{"hello", "olleh"},
		{"世界", "界世"},
		{"hello 世界", "界世 olleh"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			result := reverseString(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

// TestCapitalizeFirst tests the capitalize helper function
func TestCapitalizeFirst(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"", ""},
		{"h", "H"},
		{"hello", "Hello"},
		{"Hello", "Hello"},
		{"HELLO", "HELLO"},
		{"世界", "世界"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			result := capitalizeFirst(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

// TestConcurrentExecution tests concurrent function execution
func TestConcurrentExecution(t *testing.T) {
	fn := NewDefault()

	calls := []function.FunctionCall{
		{
			Function: fn,
			Input:    &Input{Text: "hello", Operation: "upper"},
		},
		{
			Function: fn,
			Input:    &Input{Text: "WORLD", Operation: "lower"},
		},
		{
			Function: fn,
			Input:    &Input{Text: "test", Operation: "reverse"},
		},
	}

	results := function.InvokeAll(context.Background(), calls)

	// Collect results
	var outputs []string
	for result := range results {
		require.NoError(t, result.Error)
		outputs = append(outputs, result.Output.(string))
	}

	// Should have all 3 results
	assert.Len(t, outputs, 3)
	assert.Contains(t, outputs, "HELLO")
	assert.Contains(t, outputs, "world")
	assert.Contains(t, outputs, "tset")
}
