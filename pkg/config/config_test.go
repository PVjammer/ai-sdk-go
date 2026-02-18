package config

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestNew tests creating a new config
func TestNew(t *testing.T) {
	cfg := New()
	require.NotNil(t, cfg)
	assert.NotNil(t, cfg.v)
}

// TestSet_Get tests setting and getting values
func TestSet_Get(t *testing.T) {
	cfg := New()

	cfg.Set("test.key", "value")
	assert.Equal(t, "value", cfg.Get("test.key"))
}

// TestGetString tests string getter
func TestGetString(t *testing.T) {
	cfg := New()

	cfg.Set("string_key", "test_value")
	assert.Equal(t, "test_value", cfg.GetString("string_key"))

	// Non-existent key returns empty string
	assert.Equal(t, "", cfg.GetString("nonexistent"))
}

// TestGetInt tests integer getter
func TestGetInt(t *testing.T) {
	cfg := New()

	cfg.Set("int_key", 42)
	assert.Equal(t, 42, cfg.GetInt("int_key"))

	// Non-existent key returns 0
	assert.Equal(t, 0, cfg.GetInt("nonexistent"))

	// String to int conversion
	cfg.Set("string_int", "123")
	assert.Equal(t, 123, cfg.GetInt("string_int"))
}

// TestGetInt64 tests int64 getter
func TestGetInt64(t *testing.T) {
	cfg := New()

	cfg.Set("int64_key", int64(9223372036854775807))
	assert.Equal(t, int64(9223372036854775807), cfg.GetInt64("int64_key"))
}

// TestGetFloat64 tests float64 getter
func TestGetFloat64(t *testing.T) {
	cfg := New()

	cfg.Set("float_key", 3.14159)
	assert.Equal(t, 3.14159, cfg.GetFloat64("float_key"))

	// Non-existent key returns 0.0
	assert.Equal(t, 0.0, cfg.GetFloat64("nonexistent"))
}

// TestGetBool tests boolean getter
func TestGetBool(t *testing.T) {
	cfg := New()

	cfg.Set("bool_key", true)
	assert.True(t, cfg.GetBool("bool_key"))

	cfg.Set("bool_key", false)
	assert.False(t, cfg.GetBool("bool_key"))

	// Non-existent key returns false
	assert.False(t, cfg.GetBool("nonexistent"))

	// String to bool conversion
	cfg.Set("string_bool", "true")
	assert.True(t, cfg.GetBool("string_bool"))
}

// TestGetStringSlice tests string slice getter
func TestGetStringSlice(t *testing.T) {
	cfg := New()

	slice := []string{"one", "two", "three"}
	cfg.Set("slice_key", slice)

	result := cfg.GetStringSlice("slice_key")
	assert.Equal(t, slice, result)

	// Non-existent key returns empty slice
	assert.Empty(t, cfg.GetStringSlice("nonexistent"))
}

// TestGetStringMap tests string map getter
func TestGetStringMap(t *testing.T) {
	cfg := New()

	testMap := map[string]interface{}{
		"key1": "value1",
		"key2": 42,
	}
	cfg.Set("map_key", testMap)

	result := cfg.GetStringMap("map_key")
	assert.Equal(t, "value1", result["key1"])
	assert.Equal(t, 42, result["key2"])
}

// TestIsSet tests checking if a key is set
func TestIsSet(t *testing.T) {
	cfg := New()

	cfg.Set("existing_key", "value")

	assert.True(t, cfg.IsSet("existing_key"))
	assert.False(t, cfg.IsSet("nonexistent_key"))
}

// TestDotNotation tests dot notation for nested keys
func TestDotNotation(t *testing.T) {
	cfg := New()

	cfg.Set("level1.level2.level3", "deep_value")

	assert.Equal(t, "deep_value", cfg.GetString("level1.level2.level3"))
	assert.True(t, cfg.IsSet("level1.level2.level3"))
}

// TestSub tests getting a subsection of config
func TestSub(t *testing.T) {
	cfg := New()

	cfg.Set("llm.anthropic.api_key", "test_key")
	cfg.Set("llm.anthropic.model", "claude-3")
	cfg.Set("llm.openai.api_key", "openai_key")

	// Get anthropic subsection
	anthropic := cfg.Sub("llm.anthropic")
	require.NotNil(t, anthropic)

	assert.Equal(t, "test_key", anthropic.GetString("api_key"))
	assert.Equal(t, "claude-3", anthropic.GetString("model"))

	// Subsection shouldn't see parent's other keys
	assert.Equal(t, "", anthropic.GetString("openai.api_key"))
}

// TestSub_NonExistent tests getting a non-existent subsection
func TestSub_NonExistent(t *testing.T) {
	cfg := New()

	sub := cfg.Sub("nonexistent.key")
	require.NotNil(t, sub) // Should return empty config, not nil

	assert.Equal(t, "", sub.GetString("anything"))
}

// TestAllSettings tests getting all settings
func TestAllSettings(t *testing.T) {
	cfg := New()

	cfg.Set("key1", "value1")
	cfg.Set("key2", 42)
	cfg.Set("nested.key", "nested_value")

	settings := cfg.AllSettings()

	assert.Equal(t, "value1", settings["key1"])
	assert.Equal(t, 42, settings["key2"])

	// Nested should be a map
	nested, ok := settings["nested"].(map[string]interface{})
	require.True(t, ok)
	assert.Equal(t, "nested_value", nested["key"])
}

// TestAllKeys tests getting all keys
func TestAllKeys(t *testing.T) {
	cfg := New()

	cfg.Set("key1", "value1")
	cfg.Set("key2", "value2")
	cfg.Set("nested.key", "value3")

	keys := cfg.AllKeys()

	assert.Contains(t, keys, "key1")
	assert.Contains(t, keys, "key2")
	assert.Contains(t, keys, "nested.key")
}

// TestLoadFile tests loading configuration from a file
func TestLoadFile(t *testing.T) {
	// Create temporary config file
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "test_config.yaml")

	yamlContent := `
test:
  string: "test_value"
  number: 42
  nested:
    key: "nested_value"
`

	err := os.WriteFile(configPath, []byte(yamlContent), 0644)
	require.NoError(t, err)

	// Load config
	cfg := New()
	err = cfg.LoadFile(configPath)
	require.NoError(t, err)

	// Verify values
	assert.Equal(t, "test_value", cfg.GetString("test.string"))
	assert.Equal(t, 42, cfg.GetInt("test.number"))
	assert.Equal(t, "nested_value", cfg.GetString("test.nested.key"))
}

// TestLoadFile_NonExistent tests loading a non-existent file
func TestLoadFile_NonExistent(t *testing.T) {
	cfg := New()

	err := cfg.LoadFile("/nonexistent/path/config.yaml")
	assert.Error(t, err)
}

// TestLoadFile_Merge tests merging multiple config files
func TestLoadFile_Merge(t *testing.T) {
	tmpDir := t.TempDir()

	// Create first config
	config1Path := filepath.Join(tmpDir, "config1.yaml")
	yaml1 := `
key1: "value1"
shared: "from_config1"
`
	err := os.WriteFile(config1Path, []byte(yaml1), 0644)
	require.NoError(t, err)

	// Create second config
	config2Path := filepath.Join(tmpDir, "config2.yaml")
	yaml2 := `
key2: "value2"
shared: "from_config2"
`
	err = os.WriteFile(config2Path, []byte(yaml2), 0644)
	require.NoError(t, err)

	// Load both configs
	cfg := New()
	err = cfg.LoadFile(config1Path)
	require.NoError(t, err)

	err = cfg.LoadFile(config2Path)
	require.NoError(t, err)

	// Verify merge
	assert.Equal(t, "value1", cfg.GetString("key1"))
	assert.Equal(t, "value2", cfg.GetString("key2"))
	// Second config should override first
	assert.Equal(t, "from_config2", cfg.GetString("shared"))
}

// TestExpandEnv tests environment variable expansion
func TestExpandEnv(t *testing.T) {
	// Set test environment variable
	os.Setenv("TEST_VAR", "test_value")
	defer os.Unsetenv("TEST_VAR")

	tests := []struct {
		input    string
		expected string
	}{
		{"${TEST_VAR}", "test_value"},
		{"$TEST_VAR", "test_value"},
		{"prefix_${TEST_VAR}_suffix", "prefix_test_value_suffix"},
		{"no_var", "no_var"},
		{"${NONEXISTENT}", ""},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			result := ExpandEnv(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

// TestGetStringWithEnv tests getting string with env expansion
func TestGetStringWithEnv(t *testing.T) {
	os.Setenv("API_KEY", "secret_key_123")
	defer os.Unsetenv("API_KEY")

	cfg := New()
	cfg.Set("api_key", "${API_KEY}")

	result := cfg.GetStringWithEnv("api_key")
	assert.Equal(t, "secret_key_123", result)
}

// TestConfiguredFile tests getting the config file path
func TestConfiguredFile(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "test.yaml")

	err := os.WriteFile(configPath, []byte("test: value"), 0644)
	require.NoError(t, err)

	cfg := New()
	err = cfg.LoadFile(configPath)
	require.NoError(t, err)

	assert.Equal(t, configPath, cfg.ConfiguredFile())
}

// TestGlobalConfig tests the global config instance
func TestGlobalConfig(t *testing.T) {
	// Reset global config for test isolation
	ResetGlobalConfig()

	cfg1 := GlobalConfig()
	require.NotNil(t, cfg1)

	cfg2 := GlobalConfig()
	assert.Same(t, cfg1, cfg2, "GlobalConfig should return same instance")
}

// TestSetGlobalConfig tests setting custom global config
func TestSetGlobalConfig(t *testing.T) {
	// Reset for test isolation
	ResetGlobalConfig()

	// Create custom config
	custom := New()
	custom.Set("custom_key", "custom_value")

	// Set as global
	SetGlobalConfig(custom)

	// Verify global config is our custom one
	global := GlobalConfig()
	assert.Equal(t, "custom_value", global.GetString("custom_key"))

	// Cleanup
	ResetGlobalConfig()
}

// TestEnvironmentVariables tests environment variable support
func TestEnvironmentVariables(t *testing.T) {
	// Set environment variable with AI_FUNC_ prefix
	os.Setenv("AI_FUNC_TEST_KEY", "env_value")
	defer os.Unsetenv("AI_FUNC_TEST_KEY")

	cfg := New()
	err := cfg.LoadDefaults()
	require.NoError(t, err)

	// Should be accessible with dot notation
	value := cfg.GetString("test.key")
	assert.Equal(t, "env_value", value)
}

// TestEnvironmentVariables_Override tests env var override
func TestEnvironmentVariables_Override(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, ".ai-functions.yaml")

	// Create config file
	yamlContent := `
test:
  key: "file_value"
`
	err := os.WriteFile(configPath, []byte(yamlContent), 0644)
	require.NoError(t, err)

	// Set environment variable (should override file)
	os.Setenv("AI_FUNC_TEST_KEY", "env_override")
	defer os.Unsetenv("AI_FUNC_TEST_KEY")

	// Change to tmpDir so LoadDefaults finds the config
	originalDir, _ := os.Getwd()
	os.Chdir(tmpDir)
	defer os.Chdir(originalDir)

	cfg := New()
	err = cfg.LoadDefaults()
	require.NoError(t, err)

	// Environment variable should win
	value := cfg.GetString("test.key")
	assert.Equal(t, "env_override", value)
}
