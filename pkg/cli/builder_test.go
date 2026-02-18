package cli

import (
	"bytes"
	"context"
	"reflect"
	"testing"

	"github.com/pvjammer/ai-sdk-go/internal/schema"
	"github.com/pvjammer/ai-sdk-go/pkg/function"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// MockCLIFunction is a test function with string, int, and bool parameters
type MockCLIFunction struct {
	function.Base
	lastInput   interface{}
	executeFunc func(ctx context.Context, input interface{}) (interface{}, error)
}

type MockInput struct {
	Name    string `json:"name" validate:"required" description:"User name"`
	Age     int    `json:"age" validate:"min=0,max=120" description:"User age" default:"25"`
	Verbose bool   `json:"verbose" description:"Enable verbose output" default:"false"`
}

func NewMockCLIFunction() *MockCLIFunction {
	return &MockCLIFunction{
		Base: function.NewBase(function.NilConfig{}),
	}
}

func (f *MockCLIFunction) Execute(ctx context.Context, input interface{}) (interface{}, error) {
	// Use custom execute function if set
	if f.executeFunc != nil {
		return f.executeFunc(ctx, input)
	}

	// For CLI testing, input will be map[string]interface{}
	// Store it for verification
	f.lastInput = input

	// Return a simple result
	inputMap := input.(map[string]interface{})
	return map[string]interface{}{
		"message": "Processed successfully",
		"input":   inputMap,
	}, nil
}

func (f *MockCLIFunction) InputSchema() *function.Schema {
	return schema.MustGenerateSchema(reflect.TypeOf(MockInput{}))
}

func (f *MockCLIFunction) Metadata() *function.Metadata {
	return function.NewMetadata("MockCLIFunction").
		WithCLIName("mock").
		WithCLIDescription("A mock function for testing CLI generation").
		WithCategory("test")
}

// TestAsCommand_Creation tests that AsCommand creates a valid command
func TestAsCommand_Creation(t *testing.T) {
	fn := NewMockCLIFunction()
	cmd := AsCommand(fn)

	require.NotNil(t, cmd)
	assert.Equal(t, "mock", cmd.Use)
	assert.Equal(t, "A mock function for testing CLI generation", cmd.Short)
	assert.NotNil(t, cmd.RunE)
}

// TestAsCommand_Flags tests that flags are generated from schema
func TestAsCommand_Flags(t *testing.T) {
	fn := NewMockCLIFunction()
	cmd := AsCommand(fn)

	// Check that flags exist
	nameFlag := cmd.Flags().Lookup("name")
	require.NotNil(t, nameFlag, "name flag should exist")
	assert.Equal(t, "string", nameFlag.Value.Type())

	ageFlag := cmd.Flags().Lookup("age")
	require.NotNil(t, ageFlag, "age flag should exist")
	assert.Equal(t, "int", ageFlag.Value.Type())
	assert.Equal(t, "25", ageFlag.DefValue)

	verboseFlag := cmd.Flags().Lookup("verbose")
	require.NotNil(t, verboseFlag, "verbose flag should exist")
	assert.Equal(t, "bool", verboseFlag.Value.Type())
	assert.Equal(t, "false", verboseFlag.DefValue)
}

// TestAsCommand_Execution tests command execution with flags
func TestAsCommand_Execution(t *testing.T) {
	fn := NewMockCLIFunction()
	cmd := AsCommand(fn)

	// Set up command args
	cmd.SetArgs([]string{"--name", "Alice", "--age", "30", "--verbose"})

	// Capture output
	buf := new(bytes.Buffer)
	cmd.SetOut(buf)

	// Execute command
	err := cmd.Execute()
	require.NoError(t, err)

	// Verify function was called with correct input
	require.NotNil(t, fn.lastInput)
	inputMap := fn.lastInput.(map[string]interface{})
	assert.Equal(t, "Alice", inputMap["name"])
	assert.Equal(t, 30, inputMap["age"])
	assert.Equal(t, true, inputMap["verbose"])

	// Verify output contains result
	output := buf.String()
	assert.Contains(t, output, "Processed successfully")
	assert.Contains(t, output, "Alice")
}

// TestAsCommand_DefaultValues tests that default values work
func TestAsCommand_DefaultValues(t *testing.T) {
	fn := NewMockCLIFunction()
	cmd := AsCommand(fn)

	// Only set required flag
	cmd.SetArgs([]string{"--name", "Bob"})

	buf := new(bytes.Buffer)
	cmd.SetOut(buf)

	err := cmd.Execute()
	require.NoError(t, err)

	// Verify defaults were used
	inputMap := fn.lastInput.(map[string]interface{})
	assert.Equal(t, "Bob", inputMap["name"])
	assert.Equal(t, 25, inputMap["age"])     // Default from schema
	assert.Equal(t, false, inputMap["verbose"]) // Default from schema
}

// TestAsCommand_StringResult tests string result output
func TestAsCommand_StringResult(t *testing.T) {
	fn := NewMockCLIFunction()

	// Override Execute to return string
	fn.executeFunc = func(ctx context.Context, input interface{}) (interface{}, error) {
		return "Simple string result", nil
	}

	cmd := AsCommand(fn)
	cmd.SetArgs([]string{"--name", "Test"})

	buf := new(bytes.Buffer)
	cmd.SetOut(buf)

	err := cmd.Execute()
	require.NoError(t, err)

	output := buf.String()
	assert.Contains(t, output, "Simple string result")
	assert.NotContains(t, output, "{") // Should not be JSON formatted
}

// TestBuildInputFromFlags tests flag value extraction
func TestBuildInputFromFlags(t *testing.T) {
	fn := NewMockCLIFunction()
	cmd := AsCommand(fn)

	// Set flags
	cmd.Flags().Set("name", "Charlie")
	cmd.Flags().Set("age", "35")
	cmd.Flags().Set("verbose", "true")

	schema := fn.InputSchema()
	input, err := buildInputFromFlags(cmd, schema)

	require.NoError(t, err)
	assert.Equal(t, "Charlie", input["name"])
	assert.Equal(t, 35, input["age"])
	assert.Equal(t, true, input["verbose"])
}

// TestBuildInputFromFlags_EmptySchema tests with empty schema
func TestBuildInputFromFlags_EmptySchema(t *testing.T) {
	fn := NewMockCLIFunction()
	cmd := AsCommand(fn)

	input, err := buildInputFromFlags(cmd, nil)
	require.NoError(t, err)
	assert.Empty(t, input)
}

// TestBuildInputStruct tests map to struct conversion
func TestBuildInputStruct(t *testing.T) {
	inputMap := map[string]interface{}{
		"name":    "Dave",
		"age":     40,
		"verbose": true,
	}

	result, err := BuildInputStruct(inputMap, reflect.TypeOf(MockInput{}))
	require.NoError(t, err)

	input := result.(*MockInput)
	assert.Equal(t, "Dave", input.Name)
	assert.Equal(t, 40, input.Age)
	assert.Equal(t, true, input.Verbose)
}

// TestBuildInputStruct_PointerType tests with pointer to struct
func TestBuildInputStruct_PointerType(t *testing.T) {
	inputMap := map[string]interface{}{
		"name": "Eve",
		"age":  28,
	}

	result, err := BuildInputStruct(inputMap, reflect.TypeOf(&MockInput{}))
	require.NoError(t, err)

	input := result.(*MockInput)
	assert.Equal(t, "Eve", input.Name)
	assert.Equal(t, 28, input.Age)
}

// TestBuildInputStruct_InvalidType tests error handling
func TestBuildInputStruct_InvalidType(t *testing.T) {
	inputMap := map[string]interface{}{}

	_, err := BuildInputStruct(inputMap, reflect.TypeOf("string"))
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "must be a struct")
}

// TestFindFieldByJSONTag tests JSON tag lookup
func TestFindFieldByJSONTag(t *testing.T) {
	structType := reflect.TypeOf(MockInput{})

	// Test finding existing field
	field, found := findFieldByJSONTag(structType, "name")
	assert.True(t, found)
	assert.Equal(t, "Name", field.Name)

	field, found = findFieldByJSONTag(structType, "age")
	assert.True(t, found)
	assert.Equal(t, "Age", field.Name)

	// Test non-existent field
	_, found = findFieldByJSONTag(structType, "nonexistent")
	assert.False(t, found)
}

// TestAddFlag_UnsupportedType tests handling of unsupported types
func TestAddFlag_UnsupportedType(t *testing.T) {
	cmd := AsCommand(NewMockCLIFunction())

	prop := &function.Property{
		Type:        "array", // Unsupported in prototype
		Description: "Test array",
	}

	err := addFlag(cmd, "test_array", prop)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "unsupported array type")
}
