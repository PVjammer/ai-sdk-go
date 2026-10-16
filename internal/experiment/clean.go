package experiment

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
)

// Clean removes a run's results directory and the git worktrees it created, together — so
// reviewing a run's output and then discarding it is one call, not "remember to also clean up
// the worktree git still has registered."
func Clean(resultsRoot, runID string) error {
	runDir := filepath.Join(resultsRoot, runID)
	var meta RunMeta
	if err := readJSON(filepath.Join(runDir, "run.json"), &meta); err == nil {
		for _, wt := range meta.Worktrees {
			if err := removeWorktree(meta.SourceRepo, wt); err != nil {
				fmt.Fprintf(os.Stderr, "warning: %v\n", err)
			}
		}
	}
	return os.RemoveAll(runDir)
}

// CleanAll removes every run under resultsRoot.
func CleanAll(resultsRoot string) error {
	runs, err := List(resultsRoot)
	if err != nil {
		return err
	}
	for _, id := range runs {
		if err := Clean(resultsRoot, id); err != nil {
			return fmt.Errorf("clean %s: %w", id, err)
		}
	}
	return nil
}

// List returns every run ID under resultsRoot, most recent first.
func List(resultsRoot string) ([]string, error) {
	entries, err := os.ReadDir(resultsRoot)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var ids []string
	for _, e := range entries {
		if e.IsDir() {
			ids = append(ids, e.Name())
		}
	}
	sort.Sort(sort.Reverse(sort.StringSlice(ids))) // run IDs are timestamps — lexical sort is chronological
	return ids, nil
}
