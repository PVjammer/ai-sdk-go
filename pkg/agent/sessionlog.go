package agent

import (
	"crypto/rand"
	"encoding/hex"
	"sort"
	"sync"
	"time"

	"github.com/pvjammer/ai-sdk-go/pkg/llm"
)

// Entry is one node in a SessionLog's tree. See docs/design/turn-branching.md.
//
// A "leaf" anywhere in this file is just an Entry ID — the tree has no separate cursor type.
// Each Session (session.go) tracks its own current leaf; a SessionLog itself has no notion of
// "the" current position, since many spines (a trunk, any number of manually-forked branches,
// any number of turn-scratch branches) can share one log at once.
type Entry struct {
	ID         string
	ParentID   string // "" marks a root entry
	Role       string // "user" | "assistant" | "tool" | "system"
	Content    string
	ToolCalls  []llm.ToolCall // present on assistant entries that called tools
	ToolCallID string         // present on tool-result entries
	ToolName   string         // present on tool-result entries
	CreatedAt  time.Time
}

// SessionLog is an append-only, tree-structured conversation log. Ported from pi's real
// session-manager.ts model (id/parentId/branchFromId/getChildren), not its /fork command — see
// docs/design/turn-branching.md for why that distinction matters. Safe for concurrent use.
type SessionLog struct {
	mu       sync.RWMutex
	entries  map[string]Entry
	children map[string][]string // parentID -> child IDs, in append order
}

// NewSessionLog creates an empty log.
func NewSessionLog() *SessionLog {
	return &SessionLog{
		entries:  make(map[string]Entry),
		children: make(map[string][]string),
	}
}

// Append adds e as a child of parentID (""  for a new root) and returns the stored entry,
// with ID and CreatedAt filled in if they were zero. parentID is NOT required to be a "current
// leaf" of anything — any existing entry ID (or "") is a valid append point, which is what
// makes branching from an arbitrary past point ordinary rather than special.
func (l *SessionLog) Append(parentID string, e Entry) Entry {
	l.mu.Lock()
	defer l.mu.Unlock()

	if e.ID == "" {
		e.ID = newEntryID()
	}
	if e.CreatedAt.IsZero() {
		e.CreatedAt = time.Now()
	}
	e.ParentID = parentID

	l.entries[e.ID] = e
	l.children[parentID] = append(l.children[parentID], e.ID)
	return e
}

// Get returns the entry with the given ID.
func (l *SessionLog) Get(id string) (Entry, bool) {
	l.mu.RLock()
	defer l.mu.RUnlock()
	e, ok := l.entries[id]
	return e, ok
}

// Children returns parentID's direct children, in append order.
func (l *SessionLog) Children(parentID string) []Entry {
	l.mu.RLock()
	defer l.mu.RUnlock()
	ids := l.children[parentID]
	out := make([]Entry, 0, len(ids))
	for _, id := range ids {
		out = append(out, l.entries[id])
	}
	return out
}

// Path walks parentID pointers from leafID back to its root and returns them root-first — the
// live context for a model call against that leaf. leafID == "" returns an empty path.
func (l *SessionLog) Path(leafID string) []Entry {
	l.mu.RLock()
	defer l.mu.RUnlock()

	if leafID == "" {
		return nil
	}
	var rev []Entry
	for id := leafID; id != ""; {
		e, ok := l.entries[id]
		if !ok {
			break
		}
		rev = append(rev, e)
		id = e.ParentID
	}
	// rev is leaf-to-root; reverse it in place.
	for i, j := 0, len(rev)-1; i < j; i, j = i+1, j-1 {
		rev[i], rev[j] = rev[j], rev[i]
	}
	return rev
}

// Subtree returns rootID and every descendant of it, in a stable pre-order (a node before any
// of its children, children in append order) — what read_branch renders for a branch reference.
func (l *SessionLog) Subtree(rootID string) []Entry {
	l.mu.RLock()
	defer l.mu.RUnlock()

	root, ok := l.entries[rootID]
	if !ok {
		return nil
	}
	var out []Entry
	var walk func(id string)
	walk = func(id string) {
		e := l.entries[id]
		out = append(out, e)
		for _, childID := range l.children[id] {
			walk(childID)
		}
	}
	_ = root
	walk(rootID)
	return out
}

// Entries returns a snapshot copy of every entry in the log, in no particular order — suitable
// for persistence (JSON-marshal the result). The children index isn't included: LoadSessionLog
// rebuilds it from each entry's ParentID alone, so it never needs to be kept in sync on disk.
func (l *SessionLog) Entries() []Entry {
	l.mu.RLock()
	defer l.mu.RUnlock()
	out := make([]Entry, 0, len(l.entries))
	for _, e := range l.entries {
		out = append(out, e)
	}
	return out
}

// LoadSessionLog rebuilds a SessionLog from a flat entry list (e.g. read back from disk via
// Entries). The input order doesn't matter — entries are sorted by CreatedAt first, so each
// parent's children index comes back in the same creation order Append originally built it in
// (Children/Subtree's documented ordering depends on this, not just on "some" order).
func LoadSessionLog(entries []Entry) *SessionLog {
	sorted := make([]Entry, len(entries))
	copy(sorted, entries)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].CreatedAt.Before(sorted[j].CreatedAt) })

	l := NewSessionLog()
	for _, e := range sorted {
		l.entries[e.ID] = e
		l.children[e.ParentID] = append(l.children[e.ParentID], e.ID)
	}
	return l
}

// newEntryID returns a short random hex ID — not globally unique at scale, fine for a single
// log file on disk (docs/roadmap.md's Phase 0 flags true multi-writer persistence/scale as
// unresolved; this is one process reading and writing one file at a time, not that).
func newEntryID() string {
	b := make([]byte, 8)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

// EntriesToMessages converts a Path/Subtree result into the plain []llm.ChatMessage shape
// AgentBase.SeedSession expects — the tree's own bookkeeping (ID/ParentID/CreatedAt) doesn't
// travel into a model call, only the conversational content does.
func EntriesToMessages(entries []Entry) []llm.ChatMessage {
	out := make([]llm.ChatMessage, len(entries))
	for i, e := range entries {
		out[i] = llm.ChatMessage{
			Role:       e.Role,
			Content:    e.Content,
			ToolCalls:  e.ToolCalls,
			ToolCallID: e.ToolCallID,
			ToolName:   e.ToolName,
		}
	}
	return out
}

// MessageToEntry converts one llm.ChatMessage back into an Entry (ID/ParentID/CreatedAt left
// zero for Append to fill in) — used when replaying a branch's newly-produced messages back
// into the SessionLog after a turn finishes.
func MessageToEntry(m llm.ChatMessage) Entry {
	return Entry{
		Role:       m.Role,
		Content:    m.Content,
		ToolCalls:  m.ToolCalls,
		ToolCallID: m.ToolCallID,
		ToolName:   m.ToolName,
	}
}
