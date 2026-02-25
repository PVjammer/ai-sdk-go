package hooks_test

import (
	"context"
	"errors"
	"os"
	"strings"
	"testing"

	"github.com/pvjammer/ai-sdk-go/pkg/agent"
	"github.com/pvjammer/ai-sdk-go/pkg/agent/hooks"
)

func applyPostTool(h agent.AgentHooks, toolName, result string, execErr error) (string, error) {
	if h.PostTool == nil {
		return result, nil
	}
	return h.PostTool(context.Background(), toolName, result, execErr)
}

func TestLargeResult_SmallResult_PassThrough(t *testing.T) {
	h := hooks.LargeResult(100, t.TempDir())
	got, err := applyPostTool(h, "my_tool", "small result", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "small result" {
		t.Errorf("expected result unchanged, got %q", got)
	}
}

func TestLargeResult_ExecErr_PassThrough(t *testing.T) {
	// Tool errors should never be redirected to a file, even if the error
	// message string is large.
	h := hooks.LargeResult(10, t.TempDir())
	execErr := errors.New("something went wrong")
	largeErrMsg := strings.Repeat("x", 1000)
	got, err := applyPostTool(h, "my_tool", largeErrMsg, execErr)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Original (large) string passed through because execErr is set.
	if got != largeErrMsg {
		t.Errorf("expected large error string to pass through, got len=%d", len(got))
	}
}

func TestLargeResult_LargeResult_WritesFile(t *testing.T) {
	dir := t.TempDir()
	threshold := 100
	large := strings.Repeat("x", threshold+1)

	h := hooks.LargeResult(threshold, dir)
	got, err := applyPostTool(h, "search", large, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Result string should contain a file path reference, not the original data.
	if got == large {
		t.Error("expected large result to be replaced with file reference")
	}
	if !strings.Contains(got, dir) {
		t.Errorf("expected result to contain temp dir %q, got: %s", dir, got)
	}

	// Extract the file path from the result and verify the file exists with
	// the original content.
	// The format is: "[Tool result is large (N bytes) and has been saved to: /path/file\n..."
	lines := strings.SplitN(got, "\n", 2)
	path := strings.TrimPrefix(lines[0], "[Tool result is large (")
	// Find the actual path after "saved to: "
	idx := strings.Index(got, "saved to: ")
	if idx < 0 {
		t.Fatalf("could not find 'saved to: ' in result: %s", got)
	}
	filePath := strings.SplitN(got[idx+len("saved to: "):], "\n", 2)[0]

	content, readErr := os.ReadFile(filePath)
	if readErr != nil {
		t.Fatalf("could not read temp file %q: %v", filePath, readErr)
	}
	if string(content) != large {
		t.Errorf("temp file content does not match original result")
	}

	_ = path // suppress unused-var warning
}

func TestLargeResult_DefaultThreshold(t *testing.T) {
	// threshold=0 should fall back to 8192.
	h := hooks.LargeResult(0, t.TempDir())

	// 8191 chars → should pass through.
	small := strings.Repeat("x", 8191)
	got, _ := applyPostTool(h, "t", small, nil)
	if got != small {
		t.Error("8191-byte result should pass through with default threshold 8192")
	}

	// 8193 chars → should be redirected.
	large := strings.Repeat("x", 8193)
	got, _ = applyPostTool(h, "t", large, nil)
	if got == large {
		t.Error("8193-byte result should be redirected with default threshold 8192")
	}
}

func TestLargeResult_ToolNameInFilename(t *testing.T) {
	dir := t.TempDir()
	h := hooks.LargeResult(10, dir) // no exclusions
	large := strings.Repeat("x", 100)
	got, _ := applyPostTool(h, "my_tool", large, nil)

	// The temp file name should contain the tool name.
	if !strings.Contains(got, "my_tool") {
		t.Errorf("expected temp file name to contain 'my_tool', got: %s", got)
	}
}

func TestLargeResult_ExcludedTool_PassesThrough(t *testing.T) {
	// read_file is excluded: its large result must pass through unchanged (no
	// redirect, no truncation). This prevents the redirect loop where the model
	// calls read_file to access a redirected result, which itself gets redirected.
	// read_file already has max_bytes to cap its output; TokenLimit handles context.
	dir := t.TempDir()
	threshold := 20
	large := strings.Repeat("x", 100)

	h := hooks.LargeResult(threshold, dir, "read_file")
	got, err := applyPostTool(h, "read_file", large, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Must pass through unchanged — no redirect and no truncation.
	if got != large {
		t.Errorf("excluded tool result should pass through unchanged, got len=%d want len=%d", len(got), len(large))
	}
	if strings.Contains(got, "saved to:") {
		t.Error("excluded tool should not produce a 'saved to' file reference")
	}

	// No temp files should have been created in dir.
	entries, _ := os.ReadDir(dir)
	if len(entries) != 0 {
		t.Errorf("expected no temp files for excluded tool, found %d", len(entries))
	}
}

func TestLargeResult_NonExcludedTool_StillRedirects(t *testing.T) {
	// A tool not in the exclude list should still be redirected as normal.
	dir := t.TempDir()
	threshold := 20
	large := strings.Repeat("x", 100)

	h := hooks.LargeResult(threshold, dir, "read_file") // only read_file is excluded
	got, err := applyPostTool(h, "web_search", large, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !strings.Contains(got, "saved to:") {
		t.Errorf("non-excluded tool should produce a 'saved to' file reference, got: %s", got)
	}
}
