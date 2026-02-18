package hooks

import (
	"fmt"
	"testing"

	"github.com/pvjammer/ai-sdk-go/pkg/agentcontext"
)

func TestDebugCompression(t *testing.T) {
	config := &agentcontext.ExtendedConfig{
		DefaultContextID: "test",
		MaxTokens:        40,
		AutoCompress:     true,
	}
	ext := agentcontext.NewExtendedWithConfig(config)

	hook := NewSlidingWindow(5)
	ext.RegisterCompressionHook(hook)

	for i := 0; i < 20; i++ {
		ext.AddMessage(agentcontext.Message{
			Role:    "user",
			Content: "Test message", // 12 chars = 3 tokens
		})

		tokens := ext.EstimateTokens()
		msgCount := len(ext.GetMessages(0))
		needsCompress := ext.NeedsCompression()

		fmt.Printf("After message %d: %d messages, %d tokens, needs compression: %v\n",
			i+1, msgCount, tokens, needsCompress)
	}

	finalCount := len(ext.GetMessages(0))
	t.Logf("Final message count: %d", finalCount)
}
