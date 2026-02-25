// Package hooks provides ready-made AgentHooks implementations.
package hooks

import (
	"context"
	"fmt"
	"os"

	"github.com/pvjammer/ai-sdk-go/pkg/agent"
)

// LargeResult returns an AgentHooks whose PostTool hook intercepts tool results
// that exceed threshold bytes and redirects them to a temporary file.
//
// The history entry that the model sees is replaced with a short reference:
//
//	[Tool result is large (N bytes) and has been saved to: /tmp/ai-tool-name-12345.json
//	 Use the read_file tool to access it, optionally with start_line and line_count for chunked access.]
//
// The caller's UI still receives the full raw output via EventToolResult (which
// is emitted before PostTool runs), so nothing is hidden from the user.
//
// excludeTools lists tool names that should never be redirected to a file.
// Excluded tools that exceed the threshold are instead truncated with a note
// about using chunking parameters. This prevents redirect loops: read_file is
// the tool used to read redirected results, so it must never itself redirect.
//
// Parameters:
//   - threshold: byte length above which results are offloaded. 0 → default (8192).
//   - dir: directory for temp files. "" → os.TempDir().
//   - excludeTools: tool names exempt from file redirection (e.g. "read_file").
func LargeResult(threshold int, dir string, excludeTools ...string) agent.AgentHooks {
	if threshold <= 0 {
		threshold = 8192
	}
	if dir == "" {
		dir = os.TempDir()
	}

	excluded := make(map[string]bool, len(excludeTools))
	for _, t := range excludeTools {
		excluded[t] = true
	}

	return agent.AgentHooks{
		PostTool: func(ctx context.Context, toolName, result string, execErr error) (string, error) {
			// Never redirect error strings or small results.
			if execErr != nil || len(result) <= threshold {
				return result, nil
			}

			// Excluded tools pass through unchanged, even when large.
			// This prevents redirect loops: read_file is the tool the model uses
			// to access redirected results, so redirecting its output would create
			// an infinite cycle. list_files is excluded for the same reason.
			// These tools have their own built-in size limits (max_bytes, max_entries)
			// that bound their output; TokenLimit handles any context pressure.
			if excluded[toolName] {
				return result, nil
			}

			f, err := os.CreateTemp(dir, fmt.Sprintf("ai-tool-%s-*.json", toolName))
			if err != nil {
				// Can't create temp file — truncate with a note rather than failing.
				truncated := result
				if len(truncated) > threshold {
					truncated = truncated[:threshold]
				}
				return fmt.Sprintf(
					"%s\n[... truncated at %d bytes — could not write temp file: %v]",
					truncated, threshold, err,
				), nil
			}
			defer f.Close()

			if _, writeErr := f.WriteString(result); writeErr != nil {
				// Write failed — return original rather than losing data.
				return result, nil
			}

			return fmt.Sprintf(
				"[Tool result is large (%d bytes) and has been saved to: %s\n"+
					"Use the read_file tool to access it, optionally with start_line "+
					"and line_count parameters for chunked access.]",
				len(result), f.Name(),
			), nil
		},
	}
}
