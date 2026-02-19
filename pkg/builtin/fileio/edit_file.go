package fileio

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"

	"github.com/pvjammer/ai-sdk-go/internal/schema"
	"github.com/pvjammer/ai-sdk-go/internal/validation"
	"github.com/pvjammer/ai-sdk-go/pkg/function"
)

// EditFileFunction performs find-and-replace edits on a file.
//
// Modeled on Claude Code's Edit tool: the LLM provides the exact string to
// find and the replacement. By default a single unique match is required;
// set replace_all=true to replace every occurrence.
type EditFileFunction struct {
	function.Base
}

// EditFileInput defines the input for editing a file.
type EditFileInput struct {
	Path       string `json:"path" validate:"required,min=1" description:"Path of the file to edit"`
	OldString  string `json:"old_string" validate:"required" description:"Exact string to find in the file (must be unique unless replace_all=true)"`
	NewString  string `json:"new_string" description:"Replacement string (use empty string to delete the matched text)"`
	ReplaceAll bool   `json:"replace_all" description:"Replace all occurrences; if false and multiple matches exist, returns an error" default:"false"`
}

// EditFileOutput is the result of editing a file.
type EditFileOutput struct {
	Path                string `json:"path"`
	OccurrencesReplaced int    `json:"occurrences_replaced"`
}

// NewEditFile creates a new EditFileFunction with the given config.
func NewEditFile(cfg function.Config) function.AIFunction {
	return &EditFileFunction{Base: function.NewBase(cfg)}
}

// Execute implements AIFunction.
func (f *EditFileFunction) Execute(ctx context.Context, input interface{}) (interface{}, error) {
	var in *EditFileInput

	switch v := input.(type) {
	case *EditFileInput:
		in = v
	case map[string]interface{}:
		in = parseEditFileInput(v)
	default:
		return nil, function.NewExecutionError(f, "input validation", "invalid input type")
	}

	if err := validation.Validate(in); err != nil {
		return nil, function.WrapValidationError(f, err)
	}

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	absPath, err := filepath.Abs(filepath.Clean(in.Path))
	if err != nil {
		return nil, function.WrapExecutionError(f, "resolve path", err)
	}

	data, err := os.ReadFile(absPath)
	if err != nil {
		return nil, function.WrapExecutionError(f, "read file", err)
	}

	fileContent := string(data)
	count := strings.Count(fileContent, in.OldString)

	if count == 0 {
		return nil, function.NewExecutionError(f, "edit", "old_string not found in file")
	}
	if count > 1 && !in.ReplaceAll {
		return nil, function.NewExecutionError(f, "edit",
			fmt.Sprintf("%d occurrences found; set replace_all=true or provide more surrounding context to make old_string unique", count))
	}

	var updated string
	var replaced int
	if in.ReplaceAll {
		updated = strings.ReplaceAll(fileContent, in.OldString, in.NewString)
		replaced = count
	} else {
		updated = strings.Replace(fileContent, in.OldString, in.NewString, 1)
		replaced = 1
	}

	if err := os.WriteFile(absPath, []byte(updated), 0o644); err != nil {
		return nil, function.WrapExecutionError(f, "write file", err)
	}

	return &EditFileOutput{
		Path:                absPath,
		OccurrencesReplaced: replaced,
	}, nil
}

// parseEditFileInput converts a map input into EditFileInput.
func parseEditFileInput(m map[string]interface{}) *EditFileInput {
	in := &EditFileInput{
		ReplaceAll: false,
	}

	if v, ok := m["path"].(string); ok {
		in.Path = v
	}
	if v, ok := m["old_string"].(string); ok {
		in.OldString = v
	}
	if v, ok := m["new_string"].(string); ok {
		in.NewString = v
	}
	if v, ok := m["replace_all"].(bool); ok {
		in.ReplaceAll = v
	}

	return in
}

// InputSchema implements AIFunction.
func (f *EditFileFunction) InputSchema() *function.Schema {
	return schema.MustGenerateSchema(reflect.TypeOf(EditFileInput{}))
}

// Metadata implements AIFunction.
func (f *EditFileFunction) Metadata() *function.Metadata {
	return function.NewMetadata("EditFileFunction").
		WithName("edit_file").
		WithDescription("Edit a file by finding and replacing text").
		WithToolName("edit_file").
		WithToolDescription("Edits a file by replacing a specific string with new content. old_string must be unique in the file (or set replace_all=true). Set new_string to empty string to delete the matched text.").
		WithCLIName("edit-file").
		WithCLIDescription("Edit a file by finding and replacing text").
		WithCategory("filesystem").
		WithTags("file", "edit", "replace", "filesystem")
}
