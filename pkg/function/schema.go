package function

// Schema represents a JSON Schema for function input validation.
//
// This schema is used for three purposes:
//  1. Input validation before Execute()
//  2. CLI flag generation (field names, types, defaults)
//  3. LLM tool parameter definitions
//
// Schema is typically generated from Go struct tags using the
// internal/schema package.
type Schema struct {
	// Type is the JSON Schema type (always "object" for function inputs)
	Type string `json:"type"`

	// Properties maps field names to their property definitions
	Properties map[string]*Property `json:"properties"`

	// Required lists field names that are required
	Required []string `json:"required,omitempty"`

	// AdditionalProperties controls whether extra fields are allowed
	AdditionalProperties bool `json:"additionalProperties"`
}

// Property represents a single property in a JSON Schema.
//
// Properties are generated from struct field tags and describe the
// type, validation rules, and metadata for each input field.
type Property struct {
	// Type is the JSON Schema type (string, integer, number, boolean, array, object)
	Type string `json:"type"`

	// Description provides human-readable documentation
	Description string `json:"description,omitempty"`

	// Default is the default value for optional fields
	Default interface{} `json:"default,omitempty"`

	// Enum lists allowed values for the field
	Enum []interface{} `json:"enum,omitempty"`

	// Items describes array element types (for Type="array")
	Items *Property `json:"items,omitempty"`

	// Minimum/Maximum for numeric types
	Minimum *float64 `json:"minimum,omitempty"`
	Maximum *float64 `json:"maximum,omitempty"`

	// MinLength/MaxLength for string types
	MinLength *int `json:"minLength,omitempty"`
	MaxLength *int `json:"maxLength,omitempty"`

	// Pattern for regex validation (string types)
	Pattern string `json:"pattern,omitempty"`

	// Format provides hints for string formats (e.g., "email", "uri")
	Format string `json:"format,omitempty"`

	// Examples provide sample values for documentation
	Examples []interface{} `json:"examples,omitempty"`

	// -- CLI-specific metadata (not included in JSON output) --

	// Aliases provides alternative CLI flag names
	Aliases []string `json:"-"`

	// CLIHidden hides this field from CLI flags
	CLIHidden bool `json:"-"`

	// CLIFlag treats the field as a boolean flag (--flag instead of --flag=value)
	CLIFlag bool `json:"-"`
}

// NewSchema creates a new Schema with the given properties.
//
// This is a convenience constructor for manually building schemas.
// Most schemas should be generated from struct tags using
// internal/schema.GenerateSchema().
func NewSchema(properties map[string]*Property, required []string) *Schema {
	return &Schema{
		Type:                 "object",
		Properties:           properties,
		Required:             required,
		AdditionalProperties: false,
	}
}

// NewProperty creates a new Property with the given type and description.
//
// This is a convenience constructor for manually building properties.
func NewProperty(typ, description string) *Property {
	return &Property{
		Type:        typ,
		Description: description,
	}
}

// WithDefault sets the default value for the property.
func (p *Property) WithDefault(value interface{}) *Property {
	p.Default = value
	return p
}

// WithEnum sets the allowed values for the property.
func (p *Property) WithEnum(values ...interface{}) *Property {
	p.Enum = values
	return p
}

// WithExamples sets example values for the property.
func (p *Property) WithExamples(examples ...interface{}) *Property {
	p.Examples = examples
	return p
}

// WithAliases sets CLI flag aliases for the property.
func (p *Property) WithAliases(aliases ...string) *Property {
	p.Aliases = aliases
	return p
}

// HideFromCLI marks the property as hidden from CLI flags.
func (p *Property) HideFromCLI() *Property {
	p.CLIHidden = true
	return p
}

// AsFlag marks the property as a boolean flag for CLI.
func (p *Property) AsFlag() *Property {
	p.CLIFlag = true
	return p
}

// WithRange sets min/max bounds for numeric properties.
func (p *Property) WithRange(min, max float64) *Property {
	p.Minimum = &min
	p.Maximum = &max
	return p
}

// WithLengthRange sets min/max length for string properties.
func (p *Property) WithLengthRange(min, max int) *Property {
	p.MinLength = &min
	p.MaxLength = &max
	return p
}

// WithPattern sets a regex pattern for string validation.
func (p *Property) WithPattern(pattern string) *Property {
	p.Pattern = pattern
	return p
}

// WithFormat sets a format hint for string properties.
func (p *Property) WithFormat(format string) *Property {
	p.Format = format
	return p
}

// ArrayOf creates an array property with elements of the given type.
func ArrayOf(elementType string) *Property {
	return &Property{
		Type: "array",
		Items: &Property{
			Type: elementType,
		},
	}
}

// IsRequired checks if a field is in the required list.
func (s *Schema) IsRequired(fieldName string) bool {
	for _, req := range s.Required {
		if req == fieldName {
			return true
		}
	}
	return false
}

// GetProperty retrieves a property by name, or nil if not found.
func (s *Schema) GetProperty(name string) *Property {
	return s.Properties[name]
}

// AddProperty adds or updates a property in the schema.
func (s *Schema) AddProperty(name string, prop *Property) {
	if s.Properties == nil {
		s.Properties = make(map[string]*Property)
	}
	s.Properties[name] = prop
}

// AddRequired marks a field as required.
func (s *Schema) AddRequired(fieldName string) {
	if !s.IsRequired(fieldName) {
		s.Required = append(s.Required, fieldName)
	}
}
