package skill

import "context"

// sessionIDKey is the context key for the agent session ID.
type sessionIDKey struct{}

// WithSessionID returns a new context carrying the given session ID.
// Called by AgentBase.executeAgentLoop before invoking tools.
func WithSessionID(ctx context.Context, sessionID string) context.Context {
	return context.WithValue(ctx, sessionIDKey{}, sessionID)
}

// SessionIDFromCtx retrieves the session ID from a context set by WithSessionID.
// Returns an empty string if no session ID is present.
func SessionIDFromCtx(ctx context.Context) string {
	v, _ := ctx.Value(sessionIDKey{}).(string)
	return v
}
