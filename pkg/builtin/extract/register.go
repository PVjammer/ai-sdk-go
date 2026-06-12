package extract

import (
	"github.com/pvjammer/ai-sdk-go/pkg/registry"
)

func init() {
	registry.Global().RegisterWithParams("extract", New, []registry.ConfigParam{
		{Key: "llm_provider", Type: "string", Default: "", Description: "LLM provider override for LLM-based extraction (empty = use global [llm])"},
		{Key: "llm_endpoint", Type: "string", Default: "", Description: "LLM endpoint override (empty = use global [llm])"},
		{Key: "llm_model", Type: "string", Default: "", Description: "LLM model override (empty = use global [llm])"},
	})
}
