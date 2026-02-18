// Package main demonstrates a basic agent with a simple text transform tool.
package main

import (
	"context"
	"fmt"
	"os"

	"github.com/pvjammer/ai-sdk-go/pkg/agent"
	"github.com/pvjammer/ai-sdk-go/pkg/builtin/texttransform"
	"github.com/pvjammer/ai-sdk-go/pkg/llm"
)

func main() {
	// Check for Ollama URL from environment or use default
	ollamaURL := os.Getenv("OLLAMA_URL")
	if ollamaURL == "" {
		ollamaURL = "http://localhost:11434"
	}

	// Create Ollama provider
	provider, err := llm.NewOllamaProvider(ollamaURL, "llama3.2:latest")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to create Ollama provider: %v\n", err)
		fmt.Fprintln(os.Stderr, "Make sure Ollama is running: ollama serve")
		os.Exit(1)
	}

	// Create agent with default config
	config := agent.DefaultConfig()
	config.Name = "BasicAgent"
	config.SystemPrompt = "You are a helpful assistant. You have access to a text_transform tool that can transform text to uppercase, lowercase, title case, etc."
	config.ThinkingVisible = true // Show internal reasoning

	myAgent := agent.New(provider, config)

	// Register the text transform tool
	if err := myAgent.RegisterTool(texttransform.NewDefault()); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to register text transform tool: %v\n", err)
		os.Exit(1)
	}

	// Run the agent
	ctx := context.Background()
	sessionID := "demo-session"
	message := "Please transform 'hello agent world' to uppercase using your tool"

	fmt.Printf("User: %s\n\n", message)

	// Process events
	events := myAgent.Run(ctx, sessionID, message)
	for event := range events {
		switch event.Type {
		case agent.EventThinking:
			thinking := event.Data.(agent.ThinkingEvent)
			fmt.Printf("[Thinking] %s\n", thinking.Content)

		case agent.EventToolCall:
			toolCall := event.Data.(agent.ToolCallEvent)
			fmt.Printf("[Tool Call] Using %s with input: %v\n", toolCall.ToolName, toolCall.Input)

		case agent.EventToolResult:
			result := event.Data.(agent.ToolResultEvent)
			if result.Error != nil {
				fmt.Printf("[Tool Result] %s failed: %v\n", result.ToolName, result.Error)
			} else {
				fmt.Printf("[Tool Result] %s returned: %v\n", result.ToolName, result.Output)
			}

		case agent.EventResponse:
			response := event.Data.(agent.ResponseEvent)
			if response.Done {
				fmt.Printf("\nAgent: %s\n", response.Content)
			}

		case agent.EventError:
			errorEvent := event.Data.(agent.ErrorEvent)
			fmt.Fprintf(os.Stderr, "\n[Error] %s: %v\n", errorEvent.Message, errorEvent.Error)

		case agent.EventComplete:
			fmt.Println("\n[Complete] Agent finished processing")
		}
	}

	// Cleanup
	myAgent.Close()
}
