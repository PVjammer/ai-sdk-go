package calculator

import (
	"github.com/pvjammer/ai-sdk-go/pkg/registry"
)

func init() {
	registry.Global().Register("calculator", New)
}
