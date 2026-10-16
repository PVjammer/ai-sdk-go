// Tools for the "daily driver" quick-lookup experiment: many short, low-tool-call-count turns
// (find a fact in docs, a quick web lookup) rather than few large tool-heavy turns — testing
// spine growth and recall under a different workload shape than the coding tasks.
package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/pvjammer/ai-sdk-go/pkg/function"
)

// docSearchTool greps real markdown files under dir for query, returning matching lines with
// a little surrounding context — a genuine lookup over real project documentation.
type docSearchTool struct {
	function.Base
	dir string
}

func newDocSearchTool(dir string) *docSearchTool {
	return &docSearchTool{Base: function.NewBase(function.NilConfig{}), dir: dir}
}
func (t *docSearchTool) Execute(ctx context.Context, input interface{}) (interface{}, error) {
	args, _ := input.(map[string]interface{})
	query, _ := args["query"].(string)
	if query == "" {
		return nil, fmt.Errorf("query is required")
	}
	if info, err := os.Stat(t.dir); err != nil || !info.IsDir() {
		// filepath.Walk's callback gets one call with a non-nil err when the root itself can't
		// be stat'd, and the loop below treats "err != nil" as "skip this entry, not a .md
		// file" — which would otherwise make a typo'd --docs-dir return a silent, convincing
		// "no matches found" for every single query in an entire experiment run instead of
		// surfacing the actual misconfiguration.
		return nil, fmt.Errorf("docs dir %q: %w", t.dir, err)
	}

	needle := strings.ToLower(query)
	var hits []string
	err := filepath.Walk(t.dir, func(p string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() || !strings.HasSuffix(p, ".md") {
			return nil
		}
		data, err := os.ReadFile(p)
		if err != nil {
			return nil
		}
		rel, _ := filepath.Rel(t.dir, p)
		lines := strings.Split(string(data), "\n")
		for i, line := range lines {
			if strings.Contains(strings.ToLower(line), needle) {
				hits = append(hits, fmt.Sprintf("%s:%d: %s", rel, i+1, strings.TrimSpace(line)))
				if len(hits) >= 15 {
					return filepath.SkipAll
				}
			}
		}
		return nil
	})
	if err != nil && err != filepath.SkipAll {
		return nil, err
	}
	if len(hits) == 0 {
		return "no matches found", nil
	}
	return strings.Join(hits, "\n"), nil
}
func (t *docSearchTool) InputSchema() *function.Schema {
	return &function.Schema{
		Type:       "object",
		Properties: map[string]*function.Property{"query": {Type: "string", Description: "Text to search for across the project's markdown docs."}},
		Required:   []string{"query"},
	}
}
func (t *docSearchTool) Metadata() *function.Metadata {
	m := function.NewMetadata("doc_search")
	m.Description = "Search the project's markdown documentation for a term and return matching lines with file:line."
	return m
}

// webSearchTool is a SIMULATED stand-in for real web search — fixed, deterministic canned
// answers keyed by keyword, logged like lookup_fact so call counts (including unwanted
// re-calls) can be checked after the fact. Not real search: this experiment is about spine
// growth and recall shape under many quick-lookup turns, not search quality, so a real network
// call here would add cost and nondeterminism without changing what's being measured.
type webSearchTool struct {
	function.Base
	logPath string
}

var canned = map[string]string{
	"go module proxy":      "GOPROXY defaults to https://proxy.golang.org,direct; set GOPROXY=off to disable network module fetches entirely.",
	"lru cache":             "An LRU (least recently used) cache evicts the least recently accessed entry first when full; commonly implemented with a hash map plus a doubly linked list for O(1) access and eviction.",
	"context window":        "An LLM's context window is the maximum number of tokens (input + output) it can process in a single request; exceeding it requires truncation, summarization, or a sliding window.",
	"exponential backoff":   "Exponential backoff retries a failed operation with exponentially increasing delays (e.g. 1s, 2s, 4s, 8s), often with jitter added to avoid thundering-herd retries.",
	"rate limiting algorithm": "Common rate limiting algorithms: token bucket (allows bursts up to bucket size), leaky bucket (smooths output rate), and fixed/sliding window counters.",
}

func newWebSearchTool(logPath string) *webSearchTool {
	return &webSearchTool{Base: function.NewBase(function.NilConfig{}), logPath: logPath}
}
func (t *webSearchTool) Execute(ctx context.Context, input interface{}) (interface{}, error) {
	args, _ := input.(map[string]interface{})
	query, _ := args["query"].(string)
	if query == "" {
		return nil, fmt.Errorf("query is required")
	}
	if f, err := os.OpenFile(t.logPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644); err == nil {
		fmt.Fprintf(f, "%s query=%q\n", time.Now().Format(time.RFC3339Nano), query)
		f.Close()
	}
	needle := strings.ToLower(query)
	for k, v := range canned {
		if strings.Contains(needle, k) || strings.Contains(k, needle) {
			return v, nil
		}
	}
	return fmt.Sprintf("No result for %q in this (simulated) search.", query), nil
}
func (t *webSearchTool) InputSchema() *function.Schema {
	return &function.Schema{
		Type:       "object",
		Properties: map[string]*function.Property{"query": {Type: "string", Description: "Search query."}},
		Required:   []string{"query"},
	}
}
func (t *webSearchTool) Metadata() *function.Metadata {
	m := function.NewMetadata("web_search")
	m.Description = "Search the web for a short factual answer."
	return m
}
