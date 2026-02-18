// Package cli provides automatic Cobra command generation from AIFunction definitions.
//
// This package converts AIFunction metadata and schemas into CLI commands with
// auto-generated flags, help text, and execution logic.
package cli

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"reflect"
	"strings"

	"github.com/pvjammer/ai-sdk-go/pkg/function"
	"github.com/spf13/cobra"
	"golang.org/x/term"
)

// AsCommand converts an AIFunction into a Cobra command.
//
// The command name, description, and flags are generated from the function's
// metadata and input schema. The command's RunE function calls the AIFunction's
// Execute method with the parsed flag values.
//
// Supported types:
//   - string
//   - int
//   - bool
//   - []string (string slice/array)
//
// Special handling:
//   - If stdin is not a TTY and a "text" or "input" field exists, stdin is read
//     and used as the value for that field (unless explicitly set via flag)
//
// Example:
//
//	func main() {
//	    fn := mypackage.NewMyFunction()
//	    cmd := cli.AsCommand(fn)
//	    cmd.Execute()
//	}
func AsCommand(fn function.AIFunction) *cobra.Command {
	meta := fn.Metadata()
	schema := fn.InputSchema()

	cmd := &cobra.Command{
		Use:   meta.GetName("cli"),
		Short: meta.GetDescription("cli"),
		RunE: func(cmd *cobra.Command, args []string) error {
			return executeFunction(cmd, fn, schema)
		},
	}

	// Add flags from schema
	if err := addFlagsFromSchema(cmd, schema); err != nil {
		// If flag generation fails, add a placeholder error handler
		cmd.RunE = func(cmd *cobra.Command, args []string) error {
			return fmt.Errorf("command setup failed: %w", err)
		}
	}

	return cmd
}

// addFlagsFromSchema generates Cobra flags from a JSON Schema.
func addFlagsFromSchema(cmd *cobra.Command, schema *function.Schema) error {
	if schema == nil || schema.Properties == nil {
		return nil
	}

	for fieldName, prop := range schema.Properties {
		if err := addFlag(cmd, fieldName, prop); err != nil {
			return fmt.Errorf("failed to add flag %s: %w", fieldName, err)
		}
	}

	return nil
}

// addFlag adds a single flag to the command based on property type.
func addFlag(cmd *cobra.Command, name string, prop *function.Property) error {
	// Get description
	description := prop.Description
	if description == "" {
		description = fmt.Sprintf("%s parameter", name)
	}

	// Add type information to description
	description = fmt.Sprintf("%s (type: %s)", description, prop.Type)

	// Add default value to description if present
	if prop.Default != nil {
		description = fmt.Sprintf("%s [default: %v]", description, prop.Default)
	}

	// Add flag based on type
	switch prop.Type {
	case "string":
		defaultVal := ""
		if prop.Default != nil {
			if str, ok := prop.Default.(string); ok {
				defaultVal = str
			}
		}
		cmd.Flags().String(name, defaultVal, description)

	case "integer":
		defaultVal := 0
		if prop.Default != nil {
			// Handle both int and float64 from JSON unmarshaling
			switch v := prop.Default.(type) {
			case int:
				defaultVal = v
			case float64:
				defaultVal = int(v)
			}
		}
		cmd.Flags().Int(name, defaultVal, description)

	case "boolean":
		defaultVal := false
		if prop.Default != nil {
			if b, ok := prop.Default.(bool); ok {
				defaultVal = b
			}
		}
		cmd.Flags().Bool(name, defaultVal, description)

	case "array":
		// Handle string arrays
		if prop.Items != nil && prop.Items.Type == "string" {
			var defaultVal []string
			if prop.Default != nil {
				if slice, ok := prop.Default.([]string); ok {
					defaultVal = slice
				}
			}
			cmd.Flags().StringSlice(name, defaultVal, description)
		} else {
			return fmt.Errorf("unsupported array type for flag %s (only string arrays supported)", name)
		}

	default:
		// For unsupported types, return error
		return fmt.Errorf("unsupported type %s for flag %s (supported: string, integer, boolean, array of strings)", prop.Type, name)
	}

	return nil
}

// executeFunction executes the AIFunction with values from CLI flags.
func executeFunction(cmd *cobra.Command, fn function.AIFunction, schema *function.Schema) error {
	// Build input struct from flag values
	input, err := buildInputFromFlags(cmd, schema)
	if err != nil {
		return fmt.Errorf("failed to build input: %w", err)
	}

	// Execute function
	ctx := context.Background()
	result, err := fn.Execute(ctx, input)
	if err != nil {
		return fmt.Errorf("execution failed: %w", err)
	}

	// Output result as JSON
	return outputResult(cmd, result)
}

// buildInputFromFlags constructs the input map from parsed flag values.
func buildInputFromFlags(cmd *cobra.Command, schema *function.Schema) (map[string]interface{}, error) {
	input := make(map[string]interface{})

	if schema == nil || schema.Properties == nil {
		return input, nil
	}

	// Track which flags were explicitly set by the user (vs Cobra defaults).
	// Functions with config-driven defaults use this to avoid masking config
	// values with schema defaults.
	changedFlags := make(map[string]bool)

	for fieldName, prop := range schema.Properties {
		// Get flag value based on type
		var value interface{}
		var err error

		switch prop.Type {
		case "string":
			value, err = cmd.Flags().GetString(fieldName)
		case "integer":
			value, err = cmd.Flags().GetInt(fieldName)
		case "boolean":
			value, err = cmd.Flags().GetBool(fieldName)
		case "array":
			if prop.Items != nil && prop.Items.Type == "string" {
				value, err = cmd.Flags().GetStringSlice(fieldName)
			} else {
				continue // Skip unsupported array types
			}
		default:
			continue // Skip unsupported types
		}

		if err != nil {
			return nil, fmt.Errorf("failed to get flag %s: %w", fieldName, err)
		}

		input[fieldName] = value

		if flag := cmd.Flags().Lookup(fieldName); flag != nil && flag.Changed {
			changedFlags[fieldName] = true
		}
	}

	// Always set _changed_flags so parseMapInput can distinguish
	// "CLI call with no explicit flags" from "tool call without metadata".
	input["_changed_flags"] = changedFlags

	// Handle stdin for text/input fields
	if err := handleStdin(cmd, schema, input); err != nil {
		return nil, err
	}

	return input, nil
}

// outputResult outputs the function result to stdout.
func outputResult(cmd *cobra.Command, result interface{}) error {
	// Check if result is already a string
	if str, ok := result.(string); ok {
		cmd.Println(str)
		return nil
	}

	// Otherwise, output as JSON
	jsonBytes, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal result: %w", err)
	}

	cmd.Println(string(jsonBytes))
	return nil
}

// BuildInputStruct converts a map to a struct pointer using reflection.
//
// This is used to convert the CLI flag values (map[string]interface{})
// into the actual Input struct type that the function expects.
//
// Note: This is a simplified version for the prototype. The full version
// will need better type handling and error checking.
func BuildInputStruct(inputMap map[string]interface{}, structType reflect.Type) (interface{}, error) {
	// Dereference pointer type if needed
	if structType.Kind() == reflect.Ptr {
		structType = structType.Elem()
	}

	if structType.Kind() != reflect.Struct {
		return nil, fmt.Errorf("input type must be a struct, got %s", structType.Kind())
	}

	// Create new instance
	value := reflect.New(structType)
	elem := value.Elem()

	// Set fields from map
	for fieldName, fieldValue := range inputMap {
		// Find struct field (need to match json tag)
		field, found := findFieldByJSONTag(structType, fieldName)
		if !found {
			continue // Skip unknown fields
		}

		structField := elem.FieldByName(field.Name)
		if !structField.IsValid() || !structField.CanSet() {
			continue
		}

		// Set value with type conversion
		if err := setFieldValue(structField, fieldValue); err != nil {
			return nil, fmt.Errorf("failed to set field %s: %w", fieldName, err)
		}
	}

	return value.Interface(), nil
}

// findFieldByJSONTag finds a struct field by its json tag.
func findFieldByJSONTag(structType reflect.Type, jsonName string) (reflect.StructField, bool) {
	for i := 0; i < structType.NumField(); i++ {
		field := structType.Field(i)
		jsonTag := field.Tag.Get("json")
		if jsonTag == "" {
			continue
		}

		// Handle "fieldname,omitempty" format
		parts := strings.Split(jsonTag, ",")
		tagName := parts[0]

		if tagName == jsonName {
			return field, true
		}
	}

	return reflect.StructField{}, false
}

// setFieldValue sets a reflect.Value from an interface{} with type conversion.
func setFieldValue(field reflect.Value, value interface{}) error {
	if value == nil {
		return nil
	}

	valueReflect := reflect.ValueOf(value)

	// Handle type conversion
	if valueReflect.Type().AssignableTo(field.Type()) {
		field.Set(valueReflect)
		return nil
	}

	// Try converting between compatible types
	if valueReflect.Type().ConvertibleTo(field.Type()) {
		field.Set(valueReflect.Convert(field.Type()))
		return nil
	}

	return fmt.Errorf("cannot assign %s to %s", valueReflect.Type(), field.Type())
}

// handleStdin reads from stdin and sets it as the value for "text" or "input" field.
//
// Stdin is only read if:
// 1. Stdin is not a TTY (data is being piped)
// 2. A "text" or "input" field exists in the schema
// 3. The field has not been explicitly set via a flag
func handleStdin(cmd *cobra.Command, schema *function.Schema, input map[string]interface{}) error {
	// Check if stdin is a TTY
	if term.IsTerminal(int(os.Stdin.Fd())) {
		return nil // Stdin is interactive, don't read
	}

	// Find text or input field
	var fieldName string
	if _, ok := schema.Properties["text"]; ok {
		fieldName = "text"
	} else if _, ok := schema.Properties["input"]; ok {
		fieldName = "input"
	} else {
		return nil // No text/input field
	}

	// Check if field was explicitly set
	flag := cmd.Flags().Lookup(fieldName)
	if flag != nil && flag.Changed {
		return nil // Field was set via flag, don't override
	}

	// Read from stdin
	data, err := io.ReadAll(os.Stdin)
	if err != nil {
		return fmt.Errorf("failed to read from stdin: %w", err)
	}

	// Set the field value and mark it as explicitly provided
	input[fieldName] = strings.TrimSpace(string(data))
	if changed, ok := input["_changed_flags"].(map[string]bool); ok {
		changed[fieldName] = true
	}

	return nil
}
