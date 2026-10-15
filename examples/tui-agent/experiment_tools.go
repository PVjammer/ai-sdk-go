// lookup_fact is a test-only tool for comparing turn-scratch branching on vs. off (see
// docs/design/turn-branching.md's open question about whether branching meaningfully helps
// context management and whether read_branch is actually used to recover hidden detail). It
// returns a fixed, otherwise-unguessable value per topic and logs every call to a file, so a
// test script can measure — after the fact, from the log, not by trusting the transcript —
// whether the agent re-called it for a topic it should already know, instead of recalling it
// from context or retrieving it from a branch.
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

var factValues = map[string]string{
	"alpha": "7421-ALPHA",
	"beta":  "Q-9917-X",
	"gamma": "0.0734-GAMMA",
	"delta": "DLT-5582",
}

type lookupFactTool struct {
	function.Base
	logPath string
}

func newLookupFactTool(dir string) *lookupFactTool {
	return &lookupFactTool{Base: function.NewBase(function.NilConfig{}), logPath: filepath.Join(dir, "lookup_fact.log")}
}

func (t *lookupFactTool) Execute(ctx context.Context, input interface{}) (interface{}, error) {
	args, _ := input.(map[string]interface{})
	topic, _ := args["topic"].(string)
	val, ok := factValues[strings.ToLower(topic)]
	if !ok {
		return nil, fmt.Errorf("unknown topic %q; known topics: alpha, beta, gamma, delta", topic)
	}
	if f, err := os.OpenFile(t.logPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644); err == nil {
		fmt.Fprintf(f, "%s topic=%s\n", time.Now().Format(time.RFC3339Nano), strings.ToLower(topic))
		f.Close()
	}
	return val, nil
}

func (t *lookupFactTool) InputSchema() *function.Schema {
	return &function.Schema{
		Type: "object",
		Properties: map[string]*function.Property{
			"topic": {Type: "string", Description: "Which fact to look up: alpha, beta, gamma, or delta."},
		},
		Required: []string{"topic"},
	}
}
func (t *lookupFactTool) Metadata() *function.Metadata {
	m := function.NewMetadata("lookup_fact")
	m.Description = "Look up the current value of a named fact (alpha, beta, gamma, or delta) from an external system."
	return m
}
