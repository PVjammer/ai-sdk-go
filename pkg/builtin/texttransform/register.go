package texttransform

import (
	"github.com/pvjammer/ai-sdk-go/pkg/function"
	"github.com/pvjammer/ai-sdk-go/pkg/registry"
)

func init() {
	registry.Global().Register("text_transform", New)
}

// New creates a TextTransformFunction with the given config.
// This signature matches the FactoryFunc required by the registry.
func New(cfg function.Config) function.AIFunction {
	return &TextTransformFunction{
		Base: function.NewBase(cfg),
	}
}
