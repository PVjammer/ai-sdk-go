package experiment

import (
	"fmt"
	"os"
	"os/exec"
	"strings"
)

// resolveRef resolves ref (a branch, tag, or "HEAD") to an exact commit SHA in repo, once, at
// the start of a run — every case in that run, and a later rerun of the same config, builds
// from this same pinned commit rather than whatever ref happens to point to by then.
func resolveRef(repo, ref string) (string, error) {
	cmd := exec.Command("git", "-C", repo, "rev-parse", ref)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("git rev-parse %s: %w: %s", ref, err, strings.TrimSpace(string(out)))
	}
	return strings.TrimSpace(string(out)), nil
}

// addWorktree creates a fresh, detached worktree at destDir checked out to sha. destDir must
// not already exist — this package never reuses a worktree across cases or across runs; every
// case gets its own, every time, created here and only here.
func addWorktree(repo, sha, destDir string) error {
	cmd := exec.Command("git", "-C", repo, "worktree", "add", "--detach", destDir, sha)
	if out, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("git worktree add: %w: %s", err, strings.TrimSpace(string(out)))
	}
	return nil
}

// removeWorktree removes a worktree created by addWorktree and prunes its registration from
// repo. Safe to call on a path that was already removed by other means (e.g. rm -rf of a
// whole run directory) — prune cleans up the stale registration either way.
func removeWorktree(repo, destDir string) error {
	cmd := exec.Command("git", "-C", repo, "worktree", "remove", "--force", destDir)
	if out, err := cmd.CombinedOutput(); err != nil {
		pruneCmd := exec.Command("git", "-C", repo, "worktree", "prune")
		pruneCmd.Run()
		if _, statErr := os.Stat(destDir); os.IsNotExist(statErr) {
			// This is the case the doc comment above promises is safe: already gone from
			// disk, prune cleaned up the registration, nothing left to report.
			return nil
		}
		return fmt.Errorf("git worktree remove %s: %w: %s", destDir, err, strings.TrimSpace(string(out)))
	}
	return nil
}
