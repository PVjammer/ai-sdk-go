// Package fileio provides file system operations: list, read, write, and edit files.
package fileio

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"time"

	"github.com/pvjammer/ai-sdk-go/internal/schema"
	"github.com/pvjammer/ai-sdk-go/internal/validation"
	"github.com/pvjammer/ai-sdk-go/pkg/function"
)

// ListFilesFunction lists files in a directory.
type ListFilesFunction struct {
	function.Base
}

// ListFilesInput defines the input for listing files.
type ListFilesInput struct {
	Path          string `json:"path" description:"Directory path to list" default:"."`
	Recursive     bool   `json:"recursive" description:"List files recursively" default:"false"`
	Pattern       string `json:"pattern" description:"Glob pattern to filter files (e.g. '*.go', '*.md')"`
	IncludeHidden bool   `json:"include_hidden" description:"Include hidden files and directories (those starting with .)" default:"false"`
	MaxEntries    int    `json:"max_entries" description:"Maximum number of entries to return (0 = all)" default:"200"`
}

// FileEntry describes a single file or directory.
type FileEntry struct {
	Name    string `json:"name"`
	Path    string `json:"path"`
	Type    string `json:"type"`
	Size    int64  `json:"size_bytes"`
	ModTime string `json:"modified"`
}

// ListFilesOutput is the result of listing a directory.
type ListFilesOutput struct {
	Path       string      `json:"path"`
	Entries    []FileEntry `json:"entries"`
	TotalCount int         `json:"total_count"`
	Truncated  bool        `json:"truncated"`
}

// NewListFiles creates a new ListFilesFunction with the given config.
func NewListFiles(cfg function.Config) function.AIFunction {
	return &ListFilesFunction{Base: function.NewBase(cfg)}
}

// Execute implements AIFunction.
func (f *ListFilesFunction) Execute(ctx context.Context, input interface{}) (interface{}, error) {
	var in *ListFilesInput

	switch v := input.(type) {
	case *ListFilesInput:
		in = v
	case map[string]interface{}:
		in = parseListFilesInput(v)
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

	dir := in.Path
	if dir == "" {
		dir = "."
	}

	absDir, err := filepath.Abs(dir)
	if err != nil {
		return nil, function.WrapExecutionError(f, "resolve path", err)
	}

	maxEntries := in.MaxEntries
	if maxEntries == 0 {
		maxEntries = 200
	}

	var entries []FileEntry
	truncated := false

	if in.Recursive {
		err = filepath.WalkDir(absDir, func(path string, d os.DirEntry, walkErr error) error {
			if walkErr != nil {
				return nil // skip unreadable entries
			}
			if path == absDir {
				return nil // skip root itself
			}

			name := d.Name()
			if !in.IncludeHidden && strings.HasPrefix(name, ".") {
				if d.IsDir() {
					return filepath.SkipDir
				}
				return nil
			}

			if in.Pattern != "" && !d.IsDir() {
				matched, matchErr := filepath.Match(in.Pattern, name)
				if matchErr != nil || !matched {
					return matchErr
				}
			}

			if maxEntries > 0 && len(entries) >= maxEntries {
				truncated = true
				return filepath.SkipAll
			}

			entries = append(entries, buildEntry(path, d))
			return nil
		})
	} else {
		dirEntries, readErr := os.ReadDir(absDir)
		if readErr != nil {
			return nil, function.WrapExecutionError(f, "read directory", readErr)
		}

		for _, d := range dirEntries {
			name := d.Name()
			if !in.IncludeHidden && strings.HasPrefix(name, ".") {
				continue
			}

			if in.Pattern != "" && !d.IsDir() {
				matched, matchErr := filepath.Match(in.Pattern, name)
				if matchErr != nil || !matched {
					continue
				}
			}

			if maxEntries > 0 && len(entries) >= maxEntries {
				truncated = true
				break
			}

			entries = append(entries, buildEntry(filepath.Join(absDir, name), d))
		}
	}

	if err != nil {
		return nil, function.WrapExecutionError(f, "walk directory", err)
	}

	if entries == nil {
		entries = []FileEntry{}
	}

	return &ListFilesOutput{
		Path:       absDir,
		Entries:    entries,
		TotalCount: len(entries),
		Truncated:  truncated,
	}, nil
}

// buildEntry creates a FileEntry from a directory entry and its absolute path.
func buildEntry(absPath string, d os.DirEntry) FileEntry {
	entryType := "file"
	if d.IsDir() {
		entryType = "directory"
	}

	var size int64
	var modTime string
	if info, err := d.Info(); err == nil {
		size = info.Size()
		modTime = info.ModTime().UTC().Format(time.RFC3339)
	}

	return FileEntry{
		Name:    d.Name(),
		Path:    absPath,
		Type:    entryType,
		Size:    size,
		ModTime: modTime,
	}
}

// parseListFilesInput converts a map input (from CLI/LLM) into ListFilesInput.
func parseListFilesInput(m map[string]interface{}) *ListFilesInput {
	in := &ListFilesInput{
		Path:          ".",
		Recursive:     false,
		IncludeHidden: false,
		MaxEntries:    200,
	}

	if v, ok := m["path"].(string); ok {
		in.Path = v
	}
	if v, ok := m["recursive"].(bool); ok {
		in.Recursive = v
	}
	if v, ok := m["pattern"].(string); ok {
		in.Pattern = v
	}
	if v, ok := m["include_hidden"].(bool); ok {
		in.IncludeHidden = v
	}
	if v, ok := toInt(m["max_entries"]); ok {
		in.MaxEntries = v
	}

	return in
}

// InputSchema implements AIFunction.
func (f *ListFilesFunction) InputSchema() *function.Schema {
	return schema.MustGenerateSchema(reflect.TypeOf(ListFilesInput{}))
}

// Metadata implements AIFunction.
func (f *ListFilesFunction) Metadata() *function.Metadata {
	return function.NewMetadata("ListFilesFunction").
		WithName("list_files").
		WithDescription("List files in a directory").
		WithToolName("list_files").
		WithToolDescription("Lists files and directories at a given path. Supports recursive listing, glob pattern filtering (e.g. '*.go'), and hidden file inclusion.").
		WithCLIName("list-files").
		WithCLIDescription("List files and directories").
		WithCategory("filesystem").
		WithTags("file", "directory", "list", "filesystem")
}
