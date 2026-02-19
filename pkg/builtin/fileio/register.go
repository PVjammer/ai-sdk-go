package fileio

import (
	"github.com/pvjammer/ai-sdk-go/pkg/registry"
)

func init() {
	reg := registry.Global()

	reg.RegisterWithParams("list_files", NewListFiles, []registry.ConfigParam{
		{Key: "max_entries", Type: "int", Default: 200, Description: "Maximum file entries to return per call"},
	})
	reg.RegisterWithParams("read_file", NewReadFile, []registry.ConfigParam{
		{Key: "max_bytes", Type: "int", Default: 102400, Description: "Maximum bytes to read per call (default 100 KB)"},
	})
	reg.Register("write_file", NewWriteFile)
	reg.Register("edit_file", NewEditFile)

	// File-modifying tools require user approval before the agent runs them.
	reg.SetRequiresApproval("write_file", true)
	reg.SetRequiresApproval("edit_file", true)
}

// toInt converts numeric interface values (float64 from JSON, int, int64) to int.
func toInt(v interface{}) (int, bool) {
	switch n := v.(type) {
	case float64:
		return int(n), true
	case int:
		return n, true
	case int64:
		return int(n), true
	default:
		return 0, false
	}
}
