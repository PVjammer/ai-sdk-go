package extract

import (
	"github.com/pvjammer/ai-sdk-go/pkg/registry"
)

func init() {
	registry.Global().Register("extract", New)
}
