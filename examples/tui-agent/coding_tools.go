// Coding tools for the branching-vs-not comparison on a real multi-turn implementation task:
// write_file/read_file/list_files plus run_go_test, which actually shells out to the Go
// toolchain in a scoped scratch module and returns real pass/fail output — not a toy string
// tool. This is deliberately a much larger source of "scratch work" per turn (file contents,
// compiler errors, test failures) relative to final-answer length than the lookup_fact
// experiments, which is exactly the shape turn-scratch branching is meant to help with.
package main

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/pvjammer/ai-sdk-go/pkg/function"
)

// initCodingWorkdir ensures dir exists and is a buildable Go module (go.mod present), so
// run_go_test works without network access or external dependencies.
func initCodingWorkdir(dir string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	goMod := filepath.Join(dir, "go.mod")
	if _, err := os.Stat(goMod); err == nil {
		return nil // already initialized
	}
	// 1.21, not whatever's actually installed: a go.mod whose "go" line matches or exceeds the
	// installed toolchain's exact version can still trigger Go's toolchain auto-download check
	// (GOTOOLCHAIN=auto), which fails under network sandboxing even though the installed
	// toolchain would satisfy it. Comfortably below avoids that path entirely.
	return os.WriteFile(goMod, []byte("module scratch\n\ngo 1.21\n"), 0o644)
}

func safeJoin(dir, path string) (string, error) {
	full := filepath.Join(dir, path)
	if !strings.HasPrefix(full, filepath.Clean(dir)+string(os.PathSeparator)) && full != filepath.Clean(dir) {
		return "", fmt.Errorf("path %q escapes the workdir", path)
	}
	return full, nil
}

type writeFileTool struct {
	function.Base
	dir string
}

func newWriteFileTool(dir string) *writeFileTool {
	return &writeFileTool{Base: function.NewBase(function.NilConfig{}), dir: dir}
}
func (t *writeFileTool) Execute(ctx context.Context, input interface{}) (interface{}, error) {
	args, _ := input.(map[string]interface{})
	path, _ := args["path"].(string)
	content, _ := args["content"].(string)
	if path == "" {
		return nil, fmt.Errorf("path is required")
	}
	full, err := safeJoin(t.dir, path)
	if err != nil {
		return nil, err
	}
	if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
		return nil, err
	}
	if err := os.WriteFile(full, []byte(content), 0o644); err != nil {
		return nil, err
	}
	return fmt.Sprintf("wrote %d bytes to %s", len(content), path), nil
}
func (t *writeFileTool) InputSchema() *function.Schema {
	return &function.Schema{
		Type: "object",
		Properties: map[string]*function.Property{
			"path":    {Type: "string", Description: "File path relative to the project root, e.g. \"lru/lru.go\"."},
			"content": {Type: "string", Description: "Full file content to write (overwrites any existing file)."},
		},
		Required: []string{"path", "content"},
	}
}
func (t *writeFileTool) Metadata() *function.Metadata {
	m := function.NewMetadata("write_file")
	m.Description = "Write (or overwrite) a file in the project."
	return m
}

type readFileTool struct {
	function.Base
	dir string
}

func newReadFileTool(dir string) *readFileTool {
	return &readFileTool{Base: function.NewBase(function.NilConfig{}), dir: dir}
}
func (t *readFileTool) Execute(ctx context.Context, input interface{}) (interface{}, error) {
	args, _ := input.(map[string]interface{})
	path, _ := args["path"].(string)
	if path == "" {
		return nil, fmt.Errorf("path is required")
	}
	full, err := safeJoin(t.dir, path)
	if err != nil {
		return nil, err
	}
	content, err := os.ReadFile(full)
	if err != nil {
		return nil, err
	}
	return string(content), nil
}
func (t *readFileTool) InputSchema() *function.Schema {
	return &function.Schema{
		Type:       "object",
		Properties: map[string]*function.Property{"path": {Type: "string", Description: "File path relative to the project root."}},
		Required:   []string{"path"},
	}
}
func (t *readFileTool) Metadata() *function.Metadata {
	m := function.NewMetadata("read_file")
	m.Description = "Read a file's full content from the project."
	return m
}

type listFilesTool struct {
	function.Base
	dir string
}

func newListFilesTool(dir string) *listFilesTool {
	return &listFilesTool{Base: function.NewBase(function.NilConfig{}), dir: dir}
}
func (t *listFilesTool) Execute(ctx context.Context, _ interface{}) (interface{}, error) {
	var paths []string
	err := filepath.Walk(t.dir, func(p string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return nil
		}
		rel, _ := filepath.Rel(t.dir, p)
		paths = append(paths, rel)
		return nil
	})
	if err != nil {
		return nil, err
	}
	if len(paths) == 0 {
		return "(no files yet)", nil
	}
	return strings.Join(paths, "\n"), nil
}
func (t *listFilesTool) InputSchema() *function.Schema {
	return &function.Schema{Type: "object", Properties: map[string]*function.Property{}}
}
func (t *listFilesTool) Metadata() *function.Metadata {
	m := function.NewMetadata("list_files")
	m.Description = "List every file that currently exists in the project."
	return m
}

type runGoTestTool struct {
	function.Base
	dir string
}

func newRunGoTestTool(dir string) *runGoTestTool {
	return &runGoTestTool{Base: function.NewBase(function.NilConfig{}), dir: dir}
}
func (t *runGoTestTool) Execute(ctx context.Context, input interface{}) (interface{}, error) {
	args, _ := input.(map[string]interface{})
	pkg, _ := args["package"].(string)
	if pkg == "" {
		pkg = "./..."
	}
	runCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	cmd := exec.CommandContext(runCtx, "go", "test", "-v", pkg)
	cmd.Dir = t.dir
	out, err := cmd.CombinedOutput()
	result := string(out)
	if len(result) > 4000 {
		result = result[:4000] + "\n... (truncated)"
	}
	if err != nil {
		return fmt.Sprintf("FAILED (%v):\n%s", err, result), nil
	}
	return fmt.Sprintf("PASSED:\n%s", result), nil
}
func (t *runGoTestTool) InputSchema() *function.Schema {
	return &function.Schema{
		Type: "object",
		Properties: map[string]*function.Property{
			"package": {Type: "string", Description: "Package path to test, e.g. \"./lru\". Defaults to \"./...\" (everything)."},
		},
	}
}
func (t *runGoTestTool) Metadata() *function.Metadata {
	m := function.NewMetadata("run_go_test")
	m.Description = "Compile and run go test for the project (or one package), returning the real compiler/test output. " +
		"Use this to check your work — don't just assume code compiles or tests pass."
	return m
}
