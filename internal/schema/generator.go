// Package schema provides JSON Schema generation from Go struct tags.
//
// This package uses reflection to analyze Go structs and generate JSON Schema
// definitions suitable for:
//  1. Input validation
//  2. CLI flag generation
//  3. LLM tool parameter definitions
package schema

import (
	"fmt"
	"reflect"
	"strconv"
	"strings"

	"github.com/pvjammer/ai-sdk-go/pkg/function"
)

// GenerateSchema generates a JSON Schema from a Go struct type.
//
// The function uses struct tags to build the schema:
//   - json: Field name in JSON (required)
//   - validate: Validation rules (e.g., "required", "min=1", "oneof=a b c")
//   - description: Human-readable field description
//   - default: Default value as string (e.g., `default:"5"`)
//   - enum: Comma-separated list of allowed values
//   - examples: Comma-separated list of example values
//
// Example struct:
//
//	type Input struct {
//	    Text      string `json:"text" validate:"required" description:"Text to process"`
//	    MaxLength int    `json:"max_length" validate:"min=1,max=1000" description:"Maximum length" default:"100"`
//	    Operation string `json:"operation" validate:"oneof=upper lower" enum:"upper,lower" default:"upper"`
//	}
//
// The function panics if inputType is not a struct or pointer to struct.
func GenerateSchema(inputType reflect.Type) (*function.Schema, error) {
	// Dereference pointer if needed
	if inputType.Kind() == reflect.Ptr {
		inputType = inputType.Elem()
	}

	if inputType.Kind() != reflect.Struct {
		return nil, fmt.Errorf("input type must be a struct, got %s", inputType.Kind())
	}

	schema := &function.Schema{
		Type:                 "object",
		Properties:           make(map[string]*function.Property),
		Required:             []string{},
		AdditionalProperties: false,
	}

	// Iterate over struct fields
	for i := 0; i < inputType.NumField(); i++ {
		field := inputType.Field(i)

		// Skip unexported fields
		if !field.IsExported() {
			continue
		}

		// Parse JSON tag to get field name
		jsonTag := field.Tag.Get("json")
		if jsonTag == "" || jsonTag == "-" {
			continue // Skip fields without json tag or explicitly ignored
		}

		// Handle "fieldname,omitempty" format
		parts := strings.Split(jsonTag, ",")
		fieldName := parts[0]

		// Build property from field
		prop, err := buildProperty(field)
		if err != nil {
			return nil, fmt.Errorf("field %s: %w", field.Name, err)
		}

		schema.Properties[fieldName] = prop

		// Check if required
		validateTag := field.Tag.Get("validate")
		if strings.Contains(validateTag, "required") {
			schema.Required = append(schema.Required, fieldName)
		}
	}

	return schema, nil
}

// buildProperty builds a Property from a struct field.
func buildProperty(field reflect.StructField) (*function.Property, error) {
	prop := &function.Property{}

	// Determine JSON Schema type from Go type
	goType := field.Type
	if goType.Kind() == reflect.Ptr {
		goType = goType.Elem()
	}

	switch goType.Kind() {
	case reflect.String:
		prop.Type = "string"
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		prop.Type = "integer"
	case reflect.Float32, reflect.Float64:
		prop.Type = "number"
	case reflect.Bool:
		prop.Type = "boolean"
	case reflect.Slice, reflect.Array:
		prop.Type = "array"
		// Determine element type
		elemType := goType.Elem()
		prop.Items = &function.Property{}
		switch elemType.Kind() {
		case reflect.String:
			prop.Items.Type = "string"
		case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
			prop.Items.Type = "integer"
		case reflect.Float32, reflect.Float64:
			prop.Items.Type = "number"
		case reflect.Bool:
			prop.Items.Type = "boolean"
		default:
			prop.Items.Type = "object" // Complex element types
		}
	case reflect.Map, reflect.Struct:
		prop.Type = "object"
	default:
		return nil, fmt.Errorf("unsupported type: %s", goType.Kind())
	}

	// Parse tags
	if desc := field.Tag.Get("description"); desc != "" {
		prop.Description = desc
	}

	if defaultVal := field.Tag.Get("default"); defaultVal != "" {
		prop.Default = parseDefault(defaultVal, prop.Type)
	}

	if enumTag := field.Tag.Get("enum"); enumTag != "" {
		prop.Enum = parseEnum(enumTag)
	}

	if examplesTag := field.Tag.Get("examples"); examplesTag != "" {
		prop.Examples = parseExamples(examplesTag)
	}

	// Parse validation tag for constraints
	validateTag := field.Tag.Get("validate")
	if validateTag != "" {
		parseValidationConstraints(validateTag, prop)
	}

	// Parse CLI-specific tags
	if aliasesTag := field.Tag.Get("aliases"); aliasesTag != "" {
		prop.Aliases = strings.Split(aliasesTag, ",")
	}

	if cliHidden := field.Tag.Get("cli_hidden"); cliHidden == "true" {
		prop.CLIHidden = true
	}

	if cliFlag := field.Tag.Get("cli_flag"); cliFlag == "true" {
		prop.CLIFlag = true
	}

	if pattern := field.Tag.Get("pattern"); pattern != "" {
		prop.Pattern = pattern
	}

	if format := field.Tag.Get("format"); format != "" {
		prop.Format = format
	}

	return prop, nil
}

// parseDefault converts a default value string to the appropriate type.
func parseDefault(defaultStr string, jsonType string) interface{} {
	switch jsonType {
	case "string":
		return defaultStr
	case "integer":
		if val, err := strconv.ParseInt(defaultStr, 10, 64); err == nil {
			return int(val)
		}
	case "number":
		if val, err := strconv.ParseFloat(defaultStr, 64); err == nil {
			return val
		}
	case "boolean":
		if val, err := strconv.ParseBool(defaultStr); err == nil {
			return val
		}
	}
	return defaultStr // Fallback to string
}

// parseEnum parses a comma-separated enum tag into a slice.
func parseEnum(enumTag string) []interface{} {
	parts := strings.Split(enumTag, ",")
	result := make([]interface{}, len(parts))
	for i, p := range parts {
		result[i] = strings.TrimSpace(p)
	}
	return result
}

// parseExamples parses a comma-separated examples tag into a slice.
func parseExamples(examplesTag string) []interface{} {
	parts := strings.Split(examplesTag, ",")
	result := make([]interface{}, len(parts))
	for i, p := range parts {
		result[i] = strings.TrimSpace(p)
	}
	return result
}

// parseValidationConstraints parses validate tag and sets property constraints.
//
// Supported constraints:
//   - min=N: Minimum value (numeric) or length (string)
//   - max=N: Maximum value (numeric) or length (string)
//   - oneof=a b c: Enum constraint (sets Enum if not already set)
func parseValidationConstraints(validateTag string, prop *function.Property) {
	parts := strings.Split(validateTag, ",")

	for _, part := range parts {
		part = strings.TrimSpace(part)

		if strings.HasPrefix(part, "min=") {
			minStr := strings.TrimPrefix(part, "min=")
			if prop.Type == "integer" || prop.Type == "number" {
				if val, err := strconv.ParseFloat(minStr, 64); err == nil {
					prop.Minimum = &val
				}
			} else if prop.Type == "string" {
				if val, err := strconv.Atoi(minStr); err == nil {
					prop.MinLength = &val
				}
			}
		}

		if strings.HasPrefix(part, "max=") {
			maxStr := strings.TrimPrefix(part, "max=")
			if prop.Type == "integer" || prop.Type == "number" {
				if val, err := strconv.ParseFloat(maxStr, 64); err == nil {
					prop.Maximum = &val
				}
			} else if prop.Type == "string" {
				if val, err := strconv.Atoi(maxStr); err == nil {
					prop.MaxLength = &val
				}
			}
		}

		if strings.HasPrefix(part, "oneof=") {
			// Only set enum if not already set from enum tag
			if len(prop.Enum) == 0 {
				values := strings.TrimPrefix(part, "oneof=")
				// oneof uses space-separated values, not comma-separated
				parts := strings.Fields(values)
				result := make([]interface{}, len(parts))
				for i, p := range parts {
					result[i] = p
				}
				prop.Enum = result
			}
		}
	}
}

// MustGenerateSchema is like GenerateSchema but panics on error.
//
// This is useful for static initialization where the schema is known to be valid.
//
// Example:
//
//	func (f *MyFunction) InputSchema() *function.Schema {
//	    return schema.MustGenerateSchema(reflect.TypeOf(Input{}))
//	}
func MustGenerateSchema(inputType reflect.Type) *function.Schema {
	schema, err := GenerateSchema(inputType)
	if err != nil {
		panic(fmt.Sprintf("schema generation failed: %v", err))
	}
	return schema
}
