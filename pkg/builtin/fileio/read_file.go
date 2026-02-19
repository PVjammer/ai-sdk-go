package fileio

import (
	"bufio"
	"bytes"
	"context"
	"os"
	"path/filepath"
	"reflect"
	"strings"

	"github.com/pvjammer/ai-sdk-go/internal/schema"
	"github.com/pvjammer/ai-sdk-go/internal/validation"
	"github.com/pvjammer/ai-sdk-go/pkg/function"
)

// ReadFileFunction reads the contents of a file.
type ReadFileFunction struct {
	function.Base
}

// ReadFileInput defines the input for reading a file.
type ReadFileInput struct {
	Path      string `json:"path" validate:"required,min=1" description:"File path to read"`
	MaxBytes  int    `json:"max_bytes" description:"Maximum bytes to read (default 100 KB)" default:"102400"`
	StartLine int    `json:"start_line" description:"First line to return, 1-indexed (0 = beginning of file)" default:"0"`
	LineCount int    `json:"line_count" description:"Number of lines to return (0 = all remaining lines)" default:"0"`
}

// ReadFileOutput is the result of reading a file.
type ReadFileOutput struct {
	Content   string `json:"content"`
	Path      string `json:"path"`
	SizeBytes int64  `json:"size_bytes"`
	Lines     int    `json:"lines"`
	Truncated bool   `json:"truncated"`
}

// NewReadFile creates a new ReadFileFunction with the given config.
func NewReadFile(cfg function.Config) function.AIFunction {
	return &ReadFileFunction{Base: function.NewBase(cfg)}
}

// Execute implements AIFunction.
func (f *ReadFileFunction) Execute(ctx context.Context, input interface{}) (interface{}, error) {
	var in *ReadFileInput

	switch v := input.(type) {
	case *ReadFileInput:
		in = v
	case map[string]interface{}:
		in = parseReadFileInput(v, f.Config())
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

	// Get file size first.
	info, err := os.Stat(absPath)
	if err != nil {
		return nil, function.WrapExecutionError(f, "stat file", err)
	}
	fileSize := info.Size()

	maxBytes := in.MaxBytes
	if maxBytes <= 0 {
		if v := f.Config().GetInt("read_file.max_bytes"); v > 0 {
			maxBytes = v
		} else {
			maxBytes = 102400
		}
	}

	var content string
	var lineCount int
	truncated := false

	if in.StartLine > 0 || in.LineCount > 0 {
		// Line-range read.
		data, readErr := os.ReadFile(absPath)
		if readErr != nil {
			return nil, function.WrapExecutionError(f, "read file", readErr)
		}

		scanner := bufio.NewScanner(bytes.NewReader(data))
		var lines []string
		lineNum := 0

		for scanner.Scan() {
			lineNum++
			start := in.StartLine
			if start <= 0 {
				start = 1
			}
			if lineNum < start {
				continue
			}
			lines = append(lines, scanner.Text())
			if in.LineCount > 0 && len(lines) >= in.LineCount {
				// Check if there are more lines.
				if scanner.Scan() {
					truncated = true
				}
				break
			}
		}

		content = strings.Join(lines, "\n")
		lineCount = len(lines)

		// Enforce byte limit on extracted content.
		if len(content) > maxBytes {
			content = content[:maxBytes]
			truncated = true
		}
	} else {
		// Whole-file read, capped at maxBytes.
		data, readErr := os.ReadFile(absPath)
		if readErr != nil {
			return nil, function.WrapExecutionError(f, "read file", readErr)
		}

		if len(data) > maxBytes {
			data = data[:maxBytes]
			truncated = true
		}

		content = string(data)
		lineCount = strings.Count(content, "\n")
		if content != "" {
			lineCount++ // count last line if non-empty
		}
	}

	return &ReadFileOutput{
		Content:   content,
		Path:      absPath,
		SizeBytes: fileSize,
		Lines:     lineCount,
		Truncated: truncated,
	}, nil
}

// parseReadFileInput converts a map input into ReadFileInput.
func parseReadFileInput(m map[string]interface{}, cfg function.Config) *ReadFileInput {
	maxBytes := 102400
	if cfg != nil {
		if v := cfg.GetInt("read_file.max_bytes"); v > 0 {
			maxBytes = v
		}
	}

	in := &ReadFileInput{
		MaxBytes:  maxBytes,
		StartLine: 0,
		LineCount: 0,
	}

	if v, ok := m["path"].(string); ok {
		in.Path = v
	}
	if v, ok := toInt(m["max_bytes"]); ok {
		in.MaxBytes = v
	}
	if v, ok := toInt(m["start_line"]); ok {
		in.StartLine = v
	}
	if v, ok := toInt(m["line_count"]); ok {
		in.LineCount = v
	}

	return in
}

// InputSchema implements AIFunction.
func (f *ReadFileFunction) InputSchema() *function.Schema {
	return schema.MustGenerateSchema(reflect.TypeOf(ReadFileInput{}))
}

// Metadata implements AIFunction.
func (f *ReadFileFunction) Metadata() *function.Metadata {
	return function.NewMetadata("ReadFileFunction").
		WithName("read_file").
		WithDescription("Read the contents of a file").
		WithToolName("read_file").
		WithToolDescription("Reads a file and returns its contents. Supports byte limits, line-range selection (start_line + line_count), and returns file size and whether output was truncated.").
		WithCLIName("read-file").
		WithCLIDescription("Read the contents of a file").
		WithCategory("filesystem").
		WithTags("file", "read", "filesystem")
}
