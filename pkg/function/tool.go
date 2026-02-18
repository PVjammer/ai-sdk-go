package function

// AsTool converts an AIFunction to LLM tool format.
//
// Returns a tool definition compatible with OpenAI and Anthropic APIs.
// The tool definition includes:
//   - Function name (from metadata)
//   - Description (from metadata)
//   - Parameters (from input schema)
//   - Examples (from metadata, if provided)
//
// Example:
//
//	fn := texttransform.New()
//	tool := function.AsTool(fn)
//	// Use tool with OpenAI/Anthropic SDK:
//	// tools := []map[string]interface{}{tool}
//	// response := client.Chat(messages, tools)
func AsTool(fn AIFunction) map[string]interface{} {
	meta := fn.Metadata()
	schema := fn.InputSchema()

	// Convert Schema to map[string]interface{} for JSON output
	parameters := schemaToMap(schema)

	tool := map[string]interface{}{
		"type": "function",
		"function": map[string]interface{}{
			"name":        meta.GetName("tool"),
			"description": meta.GetDescription("tool"),
			"parameters":  parameters,
		},
	}

	// Add examples if provided
	if len(meta.Examples) > 0 {
		funcDef := tool["function"].(map[string]interface{})
		funcDef["examples"] = meta.Examples
	}

	return tool
}

// AsToolSlice converts multiple AIFunctions to a slice of tool definitions.
//
// Convenience function for preparing a tools list for LLM APIs.
//
// Example:
//
//	tools := function.AsToolSlice(fn1, fn2, fn3)
//	response := client.Chat(messages, tools)
func AsToolSlice(functions ...AIFunction) []map[string]interface{} {
	tools := make([]map[string]interface{}, len(functions))
	for i, fn := range functions {
		tools[i] = AsTool(fn)
	}
	return tools
}

// schemaToMap converts a Schema to a map for JSON serialization.
func schemaToMap(schema *Schema) map[string]interface{} {
	result := map[string]interface{}{
		"type":                 schema.Type,
		"additionalProperties": schema.AdditionalProperties,
	}

	if len(schema.Properties) > 0 {
		props := make(map[string]interface{})
		for name, prop := range schema.Properties {
			props[name] = propertyToMap(prop)
		}
		result["properties"] = props
	}

	if len(schema.Required) > 0 {
		result["required"] = schema.Required
	}

	return result
}

// propertyToMap converts a Property to a map for JSON serialization.
func propertyToMap(prop *Property) map[string]interface{} {
	result := map[string]interface{}{
		"type": prop.Type,
	}

	if prop.Description != "" {
		result["description"] = prop.Description
	}

	if prop.Default != nil {
		result["default"] = prop.Default
	}

	if len(prop.Enum) > 0 {
		result["enum"] = prop.Enum
	}

	if prop.Items != nil {
		result["items"] = propertyToMap(prop.Items)
	}

	if prop.Minimum != nil {
		result["minimum"] = *prop.Minimum
	}

	if prop.Maximum != nil {
		result["maximum"] = *prop.Maximum
	}

	if prop.MinLength != nil {
		result["minLength"] = *prop.MinLength
	}

	if prop.MaxLength != nil {
		result["maxLength"] = *prop.MaxLength
	}

	if prop.Pattern != "" {
		result["pattern"] = prop.Pattern
	}

	if prop.Format != "" {
		result["format"] = prop.Format
	}

	if len(prop.Examples) > 0 {
		result["examples"] = prop.Examples
	}

	return result
}
