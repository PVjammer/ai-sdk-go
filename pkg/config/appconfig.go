package config

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	toml "github.com/pelletier/go-toml/v2"
)

// AppConfig is the top-level structured configuration loaded from TOML.
type AppConfig struct {
	LLM       LLMConfig                 `toml:"llm"`
	Agent     AppAgentConfig            `toml:"agent"`
	Functions map[string]FunctionConfig `toml:"functions"`
}

// LLMConfig holds LLM provider settings.
type LLMConfig struct {
	Provider string `toml:"provider"`
	Endpoint string `toml:"endpoint"`
	Model    string `toml:"model"`
}

// AppAgentConfig holds agent behavior settings.
type AppAgentConfig struct {
	MaxIterations int     `toml:"max_iterations"`
	Temperature   float64 `toml:"temperature"`
	MaxTokens     int     `toml:"max_tokens"`
}

// FunctionConfig holds per-function settings.
//
// The three common fields (EnabledCLI, EnabledTool, RequiresApproval) are
// always present. Function-specific parameters are stored in the Params map
// and serialized flat alongside the common fields in TOML.
type FunctionConfig struct {
	EnabledCLI       bool           `toml:"enabled_cli"`
	EnabledTool      bool           `toml:"enabled_tool"`
	RequiresApproval bool           `toml:"requires_approval"`
	Params           map[string]any `toml:"-"`
}

// FunctionDefaultParam describes one configurable parameter for a function.
type FunctionDefaultParam struct {
	Key     string
	Default any
}

// FunctionDefaults describes a registered function's name and declared config parameters.
type FunctionDefaults struct {
	Name   string
	Params []FunctionDefaultParam
}

// commonFields are the reserved keys in a function config section.
var commonFields = map[string]bool{
	"enabled_cli":       true,
	"enabled_tool":      true,
	"requires_approval": true,
}

// ConfigDir returns the XDG config directory for the ai CLI, creating it if needed.
func ConfigDir() (string, error) {
	dir := os.Getenv("XDG_CONFIG_HOME")
	if dir == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", fmt.Errorf("cannot determine home directory: %w", err)
		}
		dir = filepath.Join(home, ".config")
	}
	dir = filepath.Join(dir, "ai")

	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", fmt.Errorf("cannot create config directory: %w", err)
	}
	return dir, nil
}

// ConfigPath returns the default config file path (~/.config/ai/config.toml).
func ConfigPath() (string, error) {
	dir, err := ConfigDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "config.toml"), nil
}

// LoadAppConfig reads a TOML config file into an AppConfig.
func LoadAppConfig(path string) (*AppConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("cannot read config file: %w", err)
	}

	// Unmarshal into a raw map to handle function-specific params
	var raw map[string]any
	if err := toml.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("cannot parse config file: %w", err)
	}

	cfg := &AppConfig{
		Functions: make(map[string]FunctionConfig),
	}

	// Parse LLM section
	if llmRaw, ok := raw["llm"].(map[string]any); ok {
		if v, ok := llmRaw["provider"].(string); ok {
			cfg.LLM.Provider = v
		}
		if v, ok := llmRaw["endpoint"].(string); ok {
			cfg.LLM.Endpoint = v
		}
		if v, ok := llmRaw["model"].(string); ok {
			cfg.LLM.Model = v
		}
	}

	// Parse Agent section
	if agentRaw, ok := raw["agent"].(map[string]any); ok {
		if v, ok := tomlInt(agentRaw["max_iterations"]); ok {
			cfg.Agent.MaxIterations = v
		}
		if v, ok := tomlFloat(agentRaw["temperature"]); ok {
			cfg.Agent.Temperature = v
		}
		if v, ok := tomlInt(agentRaw["max_tokens"]); ok {
			cfg.Agent.MaxTokens = v
		}
	}

	// Parse Functions sections
	if funcsRaw, ok := raw["functions"].(map[string]any); ok {
		for name, v := range funcsRaw {
			fMap, ok := v.(map[string]any)
			if !ok {
				continue
			}

			fc := FunctionConfig{
				EnabledCLI:  true,
				EnabledTool: true,
				Params:      make(map[string]any),
			}

			if v, ok := fMap["enabled_cli"].(bool); ok {
				fc.EnabledCLI = v
			}
			if v, ok := fMap["enabled_tool"].(bool); ok {
				fc.EnabledTool = v
			}
			if v, ok := fMap["requires_approval"].(bool); ok {
				fc.RequiresApproval = v
			}

			// Everything else goes into Params
			for k, val := range fMap {
				if !commonFields[k] {
					fc.Params[k] = val
				}
			}

			cfg.Functions[name] = fc
		}
	}

	return cfg, nil
}

// SaveAppConfig writes an AppConfig to a TOML file with descriptive comments.
func SaveAppConfig(path string, cfg *AppConfig) error {
	content := MarshalAppConfig(cfg)

	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("cannot create config directory: %w", err)
	}

	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		return fmt.Errorf("cannot write config file: %w", err)
	}
	return nil
}

// MarshalAppConfig produces a commented TOML string from an AppConfig.
func MarshalAppConfig(cfg *AppConfig) string {
	var b strings.Builder

	b.WriteString("# AI CLI Configuration\n")
	b.WriteString("# Generated by: ai init\n\n")

	// LLM section
	b.WriteString("# LLM provider settings\n")
	b.WriteString("[llm]\n")
	b.WriteString(fmt.Sprintf("provider = %q\n", cfg.LLM.Provider))
	b.WriteString(fmt.Sprintf("endpoint = %q\n", cfg.LLM.Endpoint))
	b.WriteString(fmt.Sprintf("model = %q\n", cfg.LLM.Model))
	b.WriteString("\n")

	// Agent section
	b.WriteString("# Agent behavior settings\n")
	b.WriteString("[agent]\n")
	b.WriteString(fmt.Sprintf("max_iterations = %d\n", cfg.Agent.MaxIterations))
	b.WriteString(fmt.Sprintf("temperature = %g\n", cfg.Agent.Temperature))
	b.WriteString(fmt.Sprintf("max_tokens = %d\n", cfg.Agent.MaxTokens))
	b.WriteString("\n")

	// Functions — sorted by name for deterministic output
	names := make([]string, 0, len(cfg.Functions))
	for name := range cfg.Functions {
		names = append(names, name)
	}
	sort.Strings(names)

	for _, name := range names {
		fc := cfg.Functions[name]

		b.WriteString(fmt.Sprintf("# Function: %s\n", name))
		b.WriteString(fmt.Sprintf("[functions.%s]\n", name))
		b.WriteString(fmt.Sprintf("enabled_cli = %t\n", fc.EnabledCLI))
		b.WriteString(fmt.Sprintf("enabled_tool = %t\n", fc.EnabledTool))
		b.WriteString(fmt.Sprintf("requires_approval = %t\n", fc.RequiresApproval))

		// Write function-specific params sorted by key
		paramKeys := make([]string, 0, len(fc.Params))
		for k := range fc.Params {
			paramKeys = append(paramKeys, k)
		}
		sort.Strings(paramKeys)

		for _, k := range paramKeys {
			v := fc.Params[k]
			b.WriteString(formatTOMLValue(k, v))
		}

		b.WriteString("\n")
	}

	return b.String()
}

// formatTOMLValue formats a key-value pair for TOML output.
func formatTOMLValue(key string, value any) string {
	switch v := value.(type) {
	case string:
		return fmt.Sprintf("%s = %q\n", key, v)
	case bool:
		return fmt.Sprintf("%s = %t\n", key, v)
	case int:
		return fmt.Sprintf("%s = %d\n", key, v)
	case int64:
		return fmt.Sprintf("%s = %d\n", key, v)
	case float64:
		return fmt.Sprintf("%s = %g\n", key, v)
	default:
		return fmt.Sprintf("%s = %q\n", key, fmt.Sprintf("%v", v))
	}
}

// DefaultAppConfig generates an AppConfig with defaults from the provided
// function defaults. This avoids an import cycle with the registry package.
func DefaultAppConfig(funcs []FunctionDefaults) *AppConfig {
	cfg := &AppConfig{
		LLM: LLMConfig{
			Provider: "ollama",
			Endpoint: "http://localhost:11434",
			Model:    "llama3.2:latest",
		},
		Agent: AppAgentConfig{
			MaxIterations: 10,
			Temperature:   0.7,
			MaxTokens:     2048,
		},
		Functions: make(map[string]FunctionConfig),
	}

	for _, fd := range funcs {
		fc := FunctionConfig{
			EnabledCLI:       true,
			EnabledTool:      true,
			RequiresApproval: false,
			Params:           make(map[string]any),
		}

		for _, p := range fd.Params {
			fc.Params[p.Key] = p.Default
		}

		cfg.Functions[fd.Name] = fc
	}

	return cfg
}

// EnsureConfig loads the config from disk, or generates a default
// if no config file exists.
func EnsureConfig(funcs []FunctionDefaults) (*AppConfig, error) {
	path, err := ConfigPath()
	if err != nil {
		return nil, err
	}

	if _, err := os.Stat(path); err == nil {
		return LoadAppConfig(path)
	}

	// No config file — generate default (but don't write it automatically)
	return DefaultAppConfig(funcs), nil
}

// BridgeToViper injects AppConfig values into an existing Viper-based Config
// so functions can continue reading via cfg.GetString("llm.ollama.endpoint"), etc.
func BridgeToViper(appCfg *AppConfig, viperCfg *Config) {
	// LLM settings — bridge to the key paths functions actually read
	viperCfg.Set("llm.provider", appCfg.LLM.Provider)
	viperCfg.Set("llm.ollama.endpoint", appCfg.LLM.Endpoint)
	viperCfg.Set("llm.ollama.model", appCfg.LLM.Model)

	// Function-specific params — bridge to keys like "websearch.backend"
	for name, fc := range appCfg.Functions {
		for k, v := range fc.Params {
			viperCfg.Set(name+"."+k, v)
		}
	}
}

// tomlInt converts a TOML-decoded numeric value to int.
func tomlInt(v any) (int, bool) {
	switch n := v.(type) {
	case int64:
		return int(n), true
	case float64:
		return int(n), true
	case int:
		return n, true
	default:
		return 0, false
	}
}

// tomlFloat converts a TOML-decoded numeric value to float64.
func tomlFloat(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case int64:
		return float64(n), true
	case int:
		return float64(n), true
	default:
		return 0, false
	}
}
