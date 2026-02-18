package registry

// ConfigParam declares a configurable parameter for a registered function.
//
// Functions declare their parameters at registration time so the config system
// can generate default config files and the TUI editor can render appropriate
// input controls.
type ConfigParam struct {
	// Key is the config key relative to the function section (e.g., "backend", "max_results")
	Key string

	// Type is the parameter type: "string", "int", "float", "bool"
	Type string

	// Default is the default value for this parameter
	Default any

	// Description is a human-readable description shown in the config editor
	Description string

	// Options lists valid values for enum/select parameters, nil if free-form
	Options []string
}
