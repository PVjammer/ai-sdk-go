package experiment

import (
	"fmt"
	"sort"
	"strings"
)

// toFlags converts a merged config map into tui-agent CLI flags, sorted by key for
// reproducible invocation (same map, same argv, every time — useful when debugging a
// transcript's exact command line). "context_window" -> "--context-window"; a bool becomes
// "--key=true"/"--key=false" (tui-agent's flag package requires the "=" form for bools, since
// "--branching false" would parse "false" as a second positional argument instead); anything
// else becomes two args, "--key", fmt.Sprint(value).
func toFlags(m map[string]interface{}) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	var args []string
	for _, k := range keys {
		flag := "--" + strings.ReplaceAll(k, "_", "-")
		switch v := m[k].(type) {
		case bool:
			args = append(args, fmt.Sprintf("%s=%v", flag, v))
		default:
			args = append(args, flag, fmt.Sprint(v))
		}
	}
	return args
}
