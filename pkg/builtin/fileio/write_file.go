package fileio

import (
	"context"
	"os"
	"path/filepath"
	"reflect"

	"github.com/pvjammer/ai-sdk-go/internal/schema"
	"github.com/pvjammer/ai-sdk-go/internal/validation"
	"github.com/pvjammer/ai-sdk-go/pkg/function"
)

// WriteFileFunction writes content to a file.
type WriteFileFunction struct {
	function.Base
}

// WriteFileInput defines the input for writing a file.
type WriteFileInput struct {
	Path          string `json:"path" validate:"required,min=1" description:"File path to write"`
	Content       string `json:"content" validate:"required" description:"Content to write to the file"`
	Mode          string `json:"mode" validate:"oneof=overwrite append" enum:"overwrite,append" default:"overwrite" description:"overwrite replaces the file; append adds content to the end"`
	CreateParents bool   `json:"create_parents" description:"Create parent directories if they do not exist" default:"true"`
}

// WriteFileOutput is the result of writing a file.
type WriteFileOutput struct {
	Path         string `json:"path"`
	BytesWritten int    `json:"bytes_written"`
	Created      bool   `json:"created"`
}

// NewWriteFile creates a new WriteFileFunction with the given config.
func NewWriteFile(cfg function.Config) function.AIFunction {
	return &WriteFileFunction{Base: function.NewBase(cfg)}
}

// Execute implements AIFunction.
func (f *WriteFileFunction) Execute(ctx context.Context, input interface{}) (interface{}, error) {
	var in *WriteFileInput

	switch v := input.(type) {
	case *WriteFileInput:
		in = v
	case map[string]interface{}:
		in = parseWriteFileInput(v)
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

	// Determine whether the file already exists before writing.
	_, statErr := os.Stat(absPath)
	isNew := os.IsNotExist(statErr)

	if in.CreateParents {
		if mkErr := os.MkdirAll(filepath.Dir(absPath), 0o755); mkErr != nil {
			return nil, function.WrapExecutionError(f, "create directories", mkErr)
		}
	}

	data := []byte(in.Content)

	if in.Mode == "append" {
		f2, openErr := os.OpenFile(absPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
		if openErr != nil {
			return nil, function.WrapExecutionError(f, "open file", openErr)
		}
		defer f2.Close()

		n, writeErr := f2.Write(data)
		if writeErr != nil {
			return nil, function.WrapExecutionError(f, "write file", writeErr)
		}

		return &WriteFileOutput{
			Path:         absPath,
			BytesWritten: n,
			Created:      isNew,
		}, nil
	}

	// Default: overwrite.
	if writeErr := os.WriteFile(absPath, data, 0o644); writeErr != nil {
		return nil, function.WrapExecutionError(f, "write file", writeErr)
	}

	return &WriteFileOutput{
		Path:         absPath,
		BytesWritten: len(data),
		Created:      isNew,
	}, nil
}

// parseWriteFileInput converts a map input into WriteFileInput.
func parseWriteFileInput(m map[string]interface{}) *WriteFileInput {
	in := &WriteFileInput{
		Mode:          "overwrite",
		CreateParents: true,
	}

	if v, ok := m["path"].(string); ok {
		in.Path = v
	}
	if v, ok := m["content"].(string); ok {
		in.Content = v
	}
	if v, ok := m["mode"].(string); ok {
		in.Mode = v
	}
	if _, ok := m["create_parents"]; ok {
		if v, ok := m["create_parents"].(bool); ok {
			in.CreateParents = v
		}
	}

	return in
}

// InputSchema implements AIFunction.
func (f *WriteFileFunction) InputSchema() *function.Schema {
	return schema.MustGenerateSchema(reflect.TypeOf(WriteFileInput{}))
}

// Metadata implements AIFunction.
func (f *WriteFileFunction) Metadata() *function.Metadata {
	return function.NewMetadata("WriteFileFunction").
		WithName("write_file").
		WithDescription("Write content to a file").
		WithToolName("write_file").
		WithToolDescription("Writes content to a file. Use mode=overwrite (default) to replace the file or mode=append to add content to the end. Parent directories are created automatically by default.").
		WithCLIName("write-file").
		WithCLIDescription("Write content to a file").
		WithCategory("filesystem").
		WithTags("file", "write", "filesystem")
}
