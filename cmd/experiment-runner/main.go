// Command experiment-runner scripts reproducible A/B runs of examples/tui-agent across a
// matrix of configs (model, branching, context size, ...) described in a YAML file — see
// experiments/configs/*.yaml and experiments/README.md.
//
//	go run ./cmd/experiment-runner run experiments/configs/repo-feature.yaml
//	go run ./cmd/experiment-runner list
//	go run ./cmd/experiment-runner report <run-id>
//	go run ./cmd/experiment-runner clean <run-id>
//	go run ./cmd/experiment-runner clean --all
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"

	"github.com/pvjammer/ai-sdk-go/internal/experiment"
)

func loadJSONFile(path string, v interface{}) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return json.Unmarshal(data, v)
}

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(1)
	}

	moduleRoot := findModuleRoot()
	resultsRoot := filepath.Join(moduleRoot, "experiments", "results")

	switch os.Args[1] {
	case "run":
		if len(os.Args) != 3 {
			fmt.Fprintln(os.Stderr, "usage: experiment-runner run <config.yaml>")
			os.Exit(1)
		}
		cmdRun(moduleRoot, resultsRoot, os.Args[2])

	case "list":
		cmdList(resultsRoot)

	case "report":
		if len(os.Args) != 3 {
			fmt.Fprintln(os.Stderr, "usage: experiment-runner report <run-id>")
			os.Exit(1)
		}
		cmdReport(resultsRoot, os.Args[2])

	case "clean":
		if len(os.Args) != 3 {
			fmt.Fprintln(os.Stderr, "usage: experiment-runner clean <run-id> | clean --all")
			os.Exit(1)
		}
		cmdClean(resultsRoot, os.Args[2])

	default:
		usage()
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, "usage: experiment-runner <run|list|report|clean> ...")
}

func cmdRun(moduleRoot, resultsRoot, configPath string) {
	cfg, err := experiment.Load(configPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load config: %v\n", err)
		os.Exit(1)
	}
	absConfig, _ := filepath.Abs(configPath)

	runID, err := experiment.Run(context.Background(), cfg, experiment.Options{
		ConfigPath:  absConfig,
		ResultsRoot: resultsRoot,
		ModuleRoot:  moduleRoot,
	})
	if runID != "" {
		fmt.Printf("run %s\n", runID)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "run: %v\n", err)
		os.Exit(1)
	}

	rep, err := experiment.BuildReport(resultsRoot, runID, cfg.Turns)
	if err != nil {
		fmt.Fprintf(os.Stderr, "build report: %v\n", err)
		os.Exit(1)
	}
	if err := experiment.WriteReport(resultsRoot, runID, rep); err != nil {
		fmt.Fprintf(os.Stderr, "write report: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("report: %s\n", filepath.Join(resultsRoot, runID, "report.md"))
}

func cmdList(resultsRoot string) {
	ids, err := experiment.List(resultsRoot)
	if err != nil {
		fmt.Fprintf(os.Stderr, "list: %v\n", err)
		os.Exit(1)
	}
	if len(ids) == 0 {
		fmt.Println("no runs yet")
		return
	}
	for _, id := range ids {
		fmt.Println(id)
	}
}

func cmdReport(resultsRoot, runID string) {
	var meta experiment.RunMeta
	metaPath := filepath.Join(resultsRoot, runID, "run.json")
	if err := loadJSONFile(metaPath, &meta); err != nil {
		fmt.Fprintf(os.Stderr, "read run.json: %v\n", err)
		os.Exit(1)
	}
	cfg, err := experiment.Load(meta.ConfigPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "re-read original config %s: %v (report needs the turn list)\n", meta.ConfigPath, err)
		os.Exit(1)
	}
	rep, err := experiment.BuildReport(resultsRoot, runID, cfg.Turns)
	if err != nil {
		fmt.Fprintf(os.Stderr, "build report: %v\n", err)
		os.Exit(1)
	}
	if err := experiment.WriteReport(resultsRoot, runID, rep); err != nil {
		fmt.Fprintf(os.Stderr, "write report: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("report: %s\n", filepath.Join(resultsRoot, runID, "report.md"))
}

func cmdClean(resultsRoot, arg string) {
	var err error
	if arg == "--all" {
		err = experiment.CleanAll(resultsRoot)
	} else {
		err = experiment.Clean(resultsRoot, arg)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "clean: %v\n", err)
		os.Exit(1)
	}
}

// findModuleRoot walks up from this file's own source location to the directory containing
// go.mod — robust to `go run` being invoked from anywhere, since os.Getwd() isn't.
func findModuleRoot() string {
	_, thisFile, _, _ := runtime.Caller(0)
	dir := filepath.Dir(thisFile)
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			fmt.Fprintln(os.Stderr, "could not find module root (no go.mod found above cmd/experiment-runner)")
			os.Exit(1)
		}
		dir = parent
	}
}
