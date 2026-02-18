package summarize

import (
	"github.com/pvjammer/ai-sdk-go/pkg/registry"
)

func init() {
	registry.Global().RegisterWithParams("summarize", New, []registry.ConfigParam{
		{Key: "max_length", Type: "int", Default: 100, Description: "Default maximum summary length in words"},
		{Key: "style", Type: "string", Default: "paragraph", Description: "Default output style", Options: []string{"paragraph", "bullets", "points"}},
		{Key: "focus", Type: "string", Default: "general", Description: "Default summary focus area", Options: []string{"general", "technical", "business", "academic"}},
		{Key: "language", Type: "string", Default: "english", Description: "Default output language"},
		{Key: "llm_provider", Type: "string", Default: "", Description: "LLM provider override (empty = use global [llm])"},
		{Key: "llm_endpoint", Type: "string", Default: "", Description: "LLM endpoint override (empty = use global [llm])"},
		{Key: "llm_model", Type: "string", Default: "", Description: "LLM model override (empty = use global [llm])"},
	})
}
