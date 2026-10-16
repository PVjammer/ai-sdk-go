// Structured per-turn metrics for --metrics-file, consumed by cmd/experiment-runner. One JSON
// line per one-shot turn, built directly from the real event stream and the real
// hooks.CompactionResult — not scraped from printed transcript text. That distinction matters:
// earlier manual analysis this project's sessions grepped the printed transcript for
// "[compacted]" and got false positives from a file the model had read whose own source happened
// to contain that literal string. Structured output can't have that failure mode.
//
// The actual type definitions live in internal/experiment (metrics_types.go), imported here,
// so this side and cmd/experiment-runner's report.go can never silently desync from each other.
package main

import (
	"encoding/json"
	"os"

	"github.com/pvjammer/ai-sdk-go/internal/experiment"
)

// appendMetrics appends one JSON line to path. A no-op if path is "" — --metrics-file is
// optional, and ordinary interactive/REPL use shouldn't have to care about it.
func appendMetrics(path string, m experiment.TurnMetric) error {
	if path == "" {
		return nil
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	return json.NewEncoder(f).Encode(m)
}
