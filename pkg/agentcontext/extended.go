package agentcontext

import (
	"fmt"
	"sort"
	"sync"
	"time"
)

// Extended is an advanced AgentContext with personalization, context switching,
// and compression hook support.
//
// It manages multiple conversation contexts (e.g., "work", "personal", "project-x"),
// stores personalization data, and automatically compresses contexts when they
// grow too large via registered compression hooks.
type Extended struct {
	mu sync.RWMutex

	// Multiple conversation contexts
	contexts      map[string][]Message // contextID -> messages
	currentCtxID  string               // Active context
	defaultCtxID  string               // Fallback context ID

	// Personalization data (shared across all contexts)
	personalization map[string]interface{}

	// Compression management
	hooks               []CompressionHook
	maxTokens           int  // Trigger compression above this
	autoCompress        bool // Automatically compress on AddMessage
	compressionLock     sync.Mutex
	compressing         bool  // Re-entrancy guard to prevent recursive compression
	lastCompressionError error // Last error from compression (for diagnostics)
}

// ExtendedConfig configures the Extended context.
type ExtendedConfig struct {
	// DefaultContextID is the initial context (default: "default")
	DefaultContextID string

	// MaxTokens triggers compression when exceeded (default: 8000)
	MaxTokens int

	// AutoCompress enables automatic compression on AddMessage (default: true)
	AutoCompress bool
}

// DefaultExtendedConfig returns sensible defaults for Extended.
func DefaultExtendedConfig() *ExtendedConfig {
	return &ExtendedConfig{
		DefaultContextID: "default",
		MaxTokens:        8000,
		AutoCompress:     true,
	}
}

// NewExtended creates a new Extended context with default configuration.
func NewExtended() *Extended {
	return NewExtendedWithConfig(nil)
}

// NewExtendedWithConfig creates a new Extended context with custom config.
func NewExtendedWithConfig(config *ExtendedConfig) *Extended {
	if config == nil {
		config = DefaultExtendedConfig()
	}

	return &Extended{
		contexts:        make(map[string][]Message),
		currentCtxID:    config.DefaultContextID,
		defaultCtxID:    config.DefaultContextID,
		personalization: make(map[string]interface{}),
		hooks:           make([]CompressionHook, 0),
		maxTokens:       config.MaxTokens,
		autoCompress:    config.AutoCompress,
	}
}

// AddMessage adds a message to the current context.
func (e *Extended) AddMessage(msg Message) {
	e.mu.Lock()

	if msg.Timestamp.IsZero() {
		msg.Timestamp = time.Now()
	}

	// Initialize context if it doesn't exist
	if _, exists := e.contexts[e.currentCtxID]; !exists {
		e.contexts[e.currentCtxID] = make([]Message, 0)
	}

	// Add message
	e.contexts[e.currentCtxID] = append(e.contexts[e.currentCtxID], msg)

	e.mu.Unlock()

	// Auto-compress if enabled, needed, and not already compressing
	// The compressing flag prevents re-entrant compression when hooks call AddMessage
	if e.autoCompress && !e.compressing && e.NeedsCompression() {
		e.compress()
	}
}

// GetMessages returns the last N messages from the current context.
// Returns a defensive copy to prevent external modification of internal state.
func (e *Extended) GetMessages(limit int) []Message {
	e.mu.RLock()
	defer e.mu.RUnlock()

	messages, exists := e.contexts[e.currentCtxID]
	if !exists {
		return []Message{}
	}

	// Determine which slice to return
	var toReturn []Message
	if limit <= 0 || limit > len(messages) {
		toReturn = messages
	} else {
		toReturn = messages[len(messages)-limit:]
	}

	// Return a defensive copy
	result := make([]Message, len(toReturn))
	copy(result, toReturn)
	return result
}

// Clear removes all messages from the current context.
func (e *Extended) Clear() {
	e.mu.Lock()
	defer e.mu.Unlock()

	e.contexts[e.currentCtxID] = make([]Message, 0)
}

// SetPersonalization stores personalization data.
//
// Personalization is shared across all contexts.
func (e *Extended) SetPersonalization(key string, value interface{}) {
	e.mu.Lock()
	defer e.mu.Unlock()

	e.personalization[key] = value
}

// GetPersonalization retrieves personalization data.
func (e *Extended) GetPersonalization(key string) interface{} {
	e.mu.RLock()
	defer e.mu.RUnlock()

	return e.personalization[key]
}

// SwitchContext changes the active conversation context.
//
// If the context doesn't exist, it will be created when the first message is added.
func (e *Extended) SwitchContext(contextID string) error {
	if contextID == "" {
		return fmt.Errorf("context ID cannot be empty")
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	e.currentCtxID = contextID
	return nil
}

// GetCurrentContextID returns the active context ID.
func (e *Extended) GetCurrentContextID() string {
	e.mu.RLock()
	defer e.mu.RUnlock()

	return e.currentCtxID
}

// EstimateTokens returns the approximate token count for the current context.
func (e *Extended) EstimateTokens() int {
	e.mu.RLock()
	defer e.mu.RUnlock()

	messages, exists := e.contexts[e.currentCtxID]
	if !exists {
		return 0
	}

	return estimateTokensForMessages(messages)
}

// NeedsCompression returns true if the current context exceeds the token limit.
func (e *Extended) NeedsCompression() bool {
	return e.EstimateTokens() > e.maxTokens
}

// RegisterCompressionHook adds a compression hook.
//
// Hooks are executed in priority order (highest first) when compression is triggered.
func (e *Extended) RegisterCompressionHook(hook CompressionHook) {
	e.mu.Lock()
	defer e.mu.Unlock()

	e.hooks = append(e.hooks, hook)

	// Sort by priority (highest first)
	sort.Slice(e.hooks, func(i, j int) bool {
		return e.hooks[i].Priority() > e.hooks[j].Priority()
	})
}

// compress runs registered compression hooks on the current context.
func (e *Extended) compress() {
	e.compressionLock.Lock()
	defer e.compressionLock.Unlock()

	// Set compressing flag to prevent re-entrant compression
	e.compressing = true
	defer func() { e.compressing = false }()

	e.mu.RLock()
	hooks := e.hooks
	e.mu.RUnlock()

	// Clear previous error
	e.lastCompressionError = nil

	// Run hooks in priority order
	for _, hook := range hooks {
		if hook.ShouldCompress(e) {
			// Compress returns a new context, but we're modifying in place
			// So we'll need a different approach
			// For now, hooks will modify the Extended context directly
			_, err := hook.Compress(e)
			if err != nil {
				// Store the last error for diagnostics
				e.lastCompressionError = fmt.Errorf("compression hook failed: %w", err)
				// Continue with other hooks - maybe another hook can compress
				continue
			}

			// Check if compression was sufficient
			if !e.NeedsCompression() {
				break
			}
		}
	}
}

// GetAllContextIDs returns all context IDs (for inspection/debugging).
func (e *Extended) GetAllContextIDs() []string {
	e.mu.RLock()
	defer e.mu.RUnlock()

	ids := make([]string, 0, len(e.contexts))
	for id := range e.contexts {
		ids = append(ids, id)
	}
	return ids
}

// GetContextMessageCount returns the number of messages in a specific context.
func (e *Extended) GetContextMessageCount(contextID string) int {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if messages, exists := e.contexts[contextID]; exists {
		return len(messages)
	}
	return 0
}

// ClearAllContexts removes all messages from all contexts.
func (e *Extended) ClearAllContexts() {
	e.mu.Lock()
	defer e.mu.Unlock()

	for id := range e.contexts {
		e.contexts[id] = make([]Message, 0)
	}
}

// RemoveContext deletes a specific context.
func (e *Extended) RemoveContext(contextID string) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if contextID == e.currentCtxID {
		return fmt.Errorf("cannot remove current context")
	}

	delete(e.contexts, contextID)
	return nil
}

// GetLastCompressionError returns the last error encountered during compression.
// Returns nil if the last compression was successful or no compression has occurred.
// Useful for diagnostics when compression doesn't reduce context as expected.
func (e *Extended) GetLastCompressionError() error {
	e.compressionLock.Lock()
	defer e.compressionLock.Unlock()

	return e.lastCompressionError
}

// estimateTokensForMessages estimates token count for a slice of messages.
func estimateTokensForMessages(messages []Message) int {
	totalChars := 0
	for _, msg := range messages {
		totalChars += len(msg.Content)

		// Add tool call content
		for _, tc := range msg.ToolCalls {
			totalChars += len(tc.Name)
			totalChars += len(fmt.Sprintf("%v", tc.Input))
			totalChars += len(fmt.Sprintf("%v", tc.Output))
		}

		// Add metadata
		for _, v := range msg.Metadata {
			totalChars += len(fmt.Sprintf("%v", v))
		}
	}

	// 1 token ≈ 4 characters (rough estimate)
	return totalChars / 4
}
