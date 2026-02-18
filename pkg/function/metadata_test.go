package function

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestMetadata_GetName(t *testing.T) {
	tests := []struct {
		name     string
		metadata *Metadata
		context  string
		want     string
	}{
		{
			name: "tool-specific name",
			metadata: NewMetadata("MyFunction").
				WithToolName("my_tool"),
			context: "tool",
			want:    "my_tool",
		},
		{
			name: "cli-specific name",
			metadata: NewMetadata("MyFunction").
				WithCLIName("my-cmd"),
			context: "cli",
			want:    "my-cmd",
		},
		{
			name: "shared name fallback",
			metadata: NewMetadata("MyFunction").
				WithName("my_function"),
			context: "tool",
			want:    "my_function",
		},
		{
			name: "derived name fallback",
			metadata: NewMetadata("MyFunction"),
			context:  "tool",
			want:     "MyFunction",
		},
		{
			name: "tool overrides shared",
			metadata: NewMetadata("MyFunction").
				WithName("my_function").
				WithToolName("my_tool"),
			context: "tool",
			want:    "my_tool",
		},
		{
			name: "cli uses shared when no cli name",
			metadata: NewMetadata("MyFunction").
				WithName("my_function"),
			context: "cli",
			want:    "my_function",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.metadata.GetName(tt.context)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestMetadata_GetDescription(t *testing.T) {
	tests := []struct {
		name     string
		metadata *Metadata
		context  string
		want     string
	}{
		{
			name: "tool-specific description",
			metadata: NewMetadata("MyFunction").
				WithToolDescription("Tool description"),
			context: "tool",
			want:    "Tool description",
		},
		{
			name: "cli-specific description",
			metadata: NewMetadata("MyFunction").
				WithCLIDescription("CLI description"),
			context: "cli",
			want:    "CLI description",
		},
		{
			name: "shared description fallback",
			metadata: NewMetadata("MyFunction").
				WithDescription("Shared description"),
			context: "tool",
			want:    "Shared description",
		},
		{
			name: "derived description fallback",
			metadata: NewMetadata("MyFunction"),
			context:  "tool",
			want:     "MyFunction function",
		},
		{
			name: "tool overrides shared",
			metadata: NewMetadata("MyFunction").
				WithDescription("Shared").
				WithToolDescription("Tool specific"),
			context: "tool",
			want:    "Tool specific",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.metadata.GetDescription(tt.context)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestMetadata_Tags(t *testing.T) {
	meta := NewMetadata("MyFunction").
		WithTags("tag1", "tag2", "tag3")

	assert.True(t, meta.HasTag("tag1"))
	assert.True(t, meta.HasTag("tag2"))
	assert.False(t, meta.HasTag("tag4"))

	assert.True(t, meta.HasAnyTag("tag1", "tag5"))
	assert.False(t, meta.HasAnyTag("tag4", "tag5"))

	assert.True(t, meta.HasAllTags("tag1", "tag2"))
	assert.False(t, meta.HasAllTags("tag1", "tag4"))
}

func TestMetadata_Examples(t *testing.T) {
	example1 := Example{
		Input: map[string]interface{}{
			"text": "hello",
		},
		Output:      "HELLO",
		Description: "Convert to uppercase",
	}

	example2 := Example{
		Input: map[string]interface{}{
			"text": "world",
		},
		Output: "WORLD",
	}

	meta := NewMetadata("MyFunction").
		WithExamples(example1).
		AddExample(example2)

	assert.Len(t, meta.Examples, 2)
	assert.Equal(t, example1, meta.Examples[0])
	assert.Equal(t, example2, meta.Examples[1])
}

func TestMetadata_CLIHidden(t *testing.T) {
	meta := NewMetadata("MyFunction").
		HideFromCLI()

	assert.True(t, meta.CLIHidden)
}

func TestMetadata_FluentAPI(t *testing.T) {
	// Test that all methods return *Metadata for chaining
	meta := NewMetadata("Test").
		WithName("test").
		WithDescription("Test function").
		WithToolName("test_tool").
		WithToolDescription("Tool desc").
		WithCLIName("test-cmd").
		WithCLIDescription("CLI desc").
		WithCategory("testing").
		WithTags("tag1", "tag2").
		HideFromCLI()

	assert.Equal(t, "test", meta.Name)
	assert.Equal(t, "Test function", meta.Description)
	assert.Equal(t, "test_tool", meta.ToolName)
	assert.Equal(t, "Tool desc", meta.ToolDescription)
	assert.Equal(t, "test-cmd", meta.CLIName)
	assert.Equal(t, "CLI desc", meta.CLIDescription)
	assert.Equal(t, "testing", meta.Category)
	assert.Equal(t, []string{"tag1", "tag2"}, meta.Tags)
	assert.True(t, meta.CLIHidden)
}
