package simple

import "github.com/pvjammer/ai-sdk-go/pkg/agentreg"

func init() {
	agentreg.Global().Register("simple", New)
}
