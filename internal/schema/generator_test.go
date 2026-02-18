package schema

import (
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Test input structs

type SimpleInput struct {
	Text string `json:"text" validate:"required" description:"Text to process"`
	Count int    `json:"count" validate:"min=1,max=10" description:"Number of iterations" default:"5"`
}

type ComplexInput struct {
	Query      string   `json:"query" validate:"required" description:"Search query"`
	MaxResults int      `json:"max_results" validate:"min=1,max=100" description:"Maximum results" default:"10"`
	Tags       []string `json:"tags" description:"Filter tags"`
	Enabled    bool     `json:"enabled" description:"Enable feature" default:"true"`
}

type InputWithEnum struct {
	Operation string `json:"operation" validate:"oneof=upper lower title" enum:"upper,lower,title" description:"Text operation" default:"upper"`
}

type InputWithExamples struct {
	Email string `json:"email" validate:"email" description:"Email address" examples:"test@example.com,user@domain.org"`
}

type InputWithCLIMetadata struct {
	Text    string `json:"text" validate:"required" description:"Text to process"`
	Hidden  string `json:"hidden" description:"Hidden field" cli_hidden:"true"`
	Verbose bool   `json:"verbose" description:"Verbose output" cli_flag:"true"`
}

func TestGenerateSchema_Simple(t *testing.T) {
	schema, err := GenerateSchema(reflect.TypeOf(SimpleInput{}))
	require.NoError(t, err)

	assert.Equal(t, "object", schema.Type)
	assert.False(t, schema.AdditionalProperties)

	// Check properties
	assert.Len(t, schema.Properties, 2)

	textProp := schema.Properties["text"]
	require.NotNil(t, textProp)
	assert.Equal(t, "string", textProp.Type)
	assert.Equal(t, "Text to process", textProp.Description)

	countProp := schema.Properties["count"]
	require.NotNil(t, countProp)
	assert.Equal(t, "integer", countProp.Type)
	assert.Equal(t, "Number of iterations", countProp.Description)
	assert.Equal(t, 5, countProp.Default)
	assert.NotNil(t, countProp.Minimum)
	assert.Equal(t, 1.0, *countProp.Minimum)
	assert.NotNil(t, countProp.Maximum)
	assert.Equal(t, 10.0, *countProp.Maximum)

	// Check required fields
	assert.Contains(t, schema.Required, "text")
	assert.NotContains(t, schema.Required, "count")
}

func TestGenerateSchema_Complex(t *testing.T) {
	schema, err := GenerateSchema(reflect.TypeOf(ComplexInput{}))
	require.NoError(t, err)

	// Check array property
	tagsProp := schema.Properties["tags"]
	require.NotNil(t, tagsProp)
	assert.Equal(t, "array", tagsProp.Type)
	require.NotNil(t, tagsProp.Items)
	assert.Equal(t, "string", tagsProp.Items.Type)

	// Check boolean property
	enabledProp := schema.Properties["enabled"]
	require.NotNil(t, enabledProp)
	assert.Equal(t, "boolean", enabledProp.Type)
	assert.Equal(t, true, enabledProp.Default)
}

func TestGenerateSchema_WithEnum(t *testing.T) {
	schema, err := GenerateSchema(reflect.TypeOf(InputWithEnum{}))
	require.NoError(t, err)

	opProp := schema.Properties["operation"]
	require.NotNil(t, opProp)
	assert.Equal(t, "string", opProp.Type)

	// Check enum from enum tag (takes precedence over validate oneof)
	assert.Equal(t, []interface{}{"upper", "lower", "title"}, opProp.Enum)
	assert.Equal(t, "upper", opProp.Default)
}

func TestGenerateSchema_WithExamples(t *testing.T) {
	schema, err := GenerateSchema(reflect.TypeOf(InputWithExamples{}))
	require.NoError(t, err)

	emailProp := schema.Properties["email"]
	require.NotNil(t, emailProp)
	assert.Equal(t, "string", emailProp.Type)
	assert.Equal(t, []interface{}{"test@example.com", "user@domain.org"}, emailProp.Examples)
}

func TestGenerateSchema_CLIMetadata(t *testing.T) {
	schema, err := GenerateSchema(reflect.TypeOf(InputWithCLIMetadata{}))
	require.NoError(t, err)

	// Check hidden field is included in schema but marked as CLI hidden
	hiddenProp := schema.Properties["hidden"]
	require.NotNil(t, hiddenProp)
	assert.True(t, hiddenProp.CLIHidden)

	// Check CLI flag
	verboseProp := schema.Properties["verbose"]
	require.NotNil(t, verboseProp)
	assert.True(t, verboseProp.CLIFlag)
}

func TestGenerateSchema_PointerType(t *testing.T) {
	// Test that pointer to struct works
	schema, err := GenerateSchema(reflect.TypeOf(&SimpleInput{}))
	require.NoError(t, err)

	assert.Equal(t, "object", schema.Type)
	assert.Len(t, schema.Properties, 2)
}

func TestGenerateSchema_InvalidType(t *testing.T) {
	// Test that non-struct type returns error
	_, err := GenerateSchema(reflect.TypeOf("string"))
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "must be a struct")
}

func TestGenerateSchema_OnlyExportedFields(t *testing.T) {
	type InputWithUnexported struct {
		Public  string `json:"public"`
		private string `json:"private"` // Should be ignored
	}

	schema, err := GenerateSchema(reflect.TypeOf(InputWithUnexported{}))
	require.NoError(t, err)

	// Should only have public field
	assert.Len(t, schema.Properties, 1)
	assert.NotNil(t, schema.Properties["public"])
	assert.Nil(t, schema.Properties["private"])
}

func TestGenerateSchema_IgnoredFields(t *testing.T) {
	type InputWithIgnored struct {
		Included string `json:"included"`
		Excluded string `json:"-"` // Should be ignored
		NoTag    string // Should be ignored (no json tag)
	}

	schema, err := GenerateSchema(reflect.TypeOf(InputWithIgnored{}))
	require.NoError(t, err)

	// Should only have included field
	assert.Len(t, schema.Properties, 1)
	assert.NotNil(t, schema.Properties["included"])
}

func TestMustGenerateSchema(t *testing.T) {
	// Should not panic for valid input
	assert.NotPanics(t, func() {
		schema := MustGenerateSchema(reflect.TypeOf(SimpleInput{}))
		assert.NotNil(t, schema)
	})

	// Should panic for invalid input
	assert.Panics(t, func() {
		MustGenerateSchema(reflect.TypeOf("string"))
	})
}

func TestParseValidationConstraints_Min(t *testing.T) {
	type Input struct {
		Number int    `json:"number" validate:"min=5"`
		Text   string `json:"text" validate:"min=3"`
	}

	schema, err := GenerateSchema(reflect.TypeOf(Input{}))
	require.NoError(t, err)

	numberProp := schema.Properties["number"]
	assert.NotNil(t, numberProp.Minimum)
	assert.Equal(t, 5.0, *numberProp.Minimum)

	textProp := schema.Properties["text"]
	assert.NotNil(t, textProp.MinLength)
	assert.Equal(t, 3, *textProp.MinLength)
}

func TestParseValidationConstraints_Max(t *testing.T) {
	type Input struct {
		Number int    `json:"number" validate:"max=100"`
		Text   string `json:"text" validate:"max=50"`
	}

	schema, err := GenerateSchema(reflect.TypeOf(Input{}))
	require.NoError(t, err)

	numberProp := schema.Properties["number"]
	assert.NotNil(t, numberProp.Maximum)
	assert.Equal(t, 100.0, *numberProp.Maximum)

	textProp := schema.Properties["text"]
	assert.NotNil(t, textProp.MaxLength)
	assert.Equal(t, 50, *textProp.MaxLength)
}

func TestParseValidationConstraints_OneOf(t *testing.T) {
	type Input struct {
		Op string `json:"op" validate:"oneof=add sub mul"`
	}

	schema, err := GenerateSchema(reflect.TypeOf(Input{}))
	require.NoError(t, err)

	opProp := schema.Properties["op"]
	// oneof should set Enum
	assert.Equal(t, []interface{}{"add", "sub", "mul"}, opProp.Enum)
}
