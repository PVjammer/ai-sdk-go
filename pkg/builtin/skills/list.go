package skills

import (
	"context"
	"fmt"
	"reflect"

	"github.com/pvjammer/ai-sdk-go/internal/schema"
	"github.com/pvjammer/ai-sdk-go/pkg/function"
	"github.com/pvjammer/ai-sdk-go/pkg/skill"
)

// ListSkillsFunction lists all discovered skills with their descriptions.
type ListSkillsFunction struct {
	function.Base
	loader *skill.SkillLoader
}

// ListInput is empty — list_skills takes no arguments.
type ListInput struct{}

// SkillSummary is one entry in the list_skills output.
type SkillSummary struct {
	Name         string `json:"name"`
	Description  string `json:"description"`
	Type         string `json:"type"`
	ArgumentHint string `json:"argument_hint,omitempty"`
}

func newListSkills(loader *skill.SkillLoader, cfg function.Config) function.AIFunction {
	return &ListSkillsFunction{
		Base:   function.NewBase(cfg),
		loader: loader,
	}
}

// Execute implements the AIFunction interface.
func (f *ListSkillsFunction) Execute(_ context.Context, _ interface{}) (interface{}, error) {
	if f.loader == nil {
		return nil, fmt.Errorf("list_skills: skill loader not initialized")
	}

	catalog := f.loader.Catalog()
	summaries := make([]SkillSummary, 0, len(catalog))
	for _, r := range catalog {
		summaries = append(summaries, SkillSummary{
			Name:         r.Frontmatter.Name,
			Description:  r.Frontmatter.Description,
			Type:         string(r.Type),
			ArgumentHint: r.Frontmatter.ArgumentHint,
		})
	}

	return summaries, nil
}

// InputSchema implements the AIFunction interface.
func (f *ListSkillsFunction) InputSchema() *function.Schema {
	return schema.MustGenerateSchema(reflect.TypeOf(ListInput{}))
}

// Metadata implements the AIFunction interface.
func (f *ListSkillsFunction) Metadata() *function.Metadata {
	return function.NewMetadata("ListSkillsFunction").
		WithName("list_skills").
		WithToolName("list_skills").
		WithToolDescription("List all available skills with their descriptions. Use to discover skills when the user asks what you can do.").
		WithCLIName("skills").
		WithCLIDescription("List available skills").
		WithCategory("skills").
		WithTags("skills", "meta")
}
