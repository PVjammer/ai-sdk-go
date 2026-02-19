// Package calculator provides a mathematical expression evaluator.
//
// LLMs are unreliable at arithmetic; this tool offloads numeric computation
// to the expr-lang/expr evaluator (memory-safe, guaranteed termination).
package calculator

import (
	"context"
	"fmt"
	"math"
	"reflect"

	"github.com/expr-lang/expr"
	"github.com/pvjammer/ai-sdk-go/internal/schema"
	"github.com/pvjammer/ai-sdk-go/internal/validation"
	"github.com/pvjammer/ai-sdk-go/pkg/function"
)

// CalculatorFunction evaluates mathematical expressions.
type CalculatorFunction struct {
	function.Base
}

// Input defines the input schema for the calculator.
type Input struct {
	Expression string `json:"expression" validate:"required,min=1" description:"Math expression to evaluate. Use pow(base, exp) for exponentiation. Available functions: sqrt, abs, floor, ceil, round, log (base10), ln, log2, sin, cos, tan, pow, min, max. Constants: pi, e. Example: 'sqrt(144) + pow(2, 10)'"`
}

// Output is the result of evaluating an expression.
type Output struct {
	Expression string  `json:"expression"`
	Result     float64 `json:"result"`
	ResultStr  string  `json:"result_string"`
}

// mathEnv is the fixed evaluation environment with all supported math functions
// and constants.
//
// Functions accept `any` arguments so that expr-lang can call them via
// reflection regardless of whether integer or float literals are used in the
// expression (e.g. sqrt(144) where 144 is an int literal).
var mathEnv = map[string]interface{}{
	"pi":    math.Pi,
	"e":     math.E,
	"sqrt":  f1(math.Sqrt),
	"abs":   f1(math.Abs),
	"floor": f1(math.Floor),
	"ceil":  f1(math.Ceil),
	"round": f1(math.Round),
	"log":   f1(math.Log10),
	"ln":    f1(math.Log),
	"log2":  f1(math.Log2),
	"sin":   f1(math.Sin),
	"cos":   f1(math.Cos),
	"tan":   f1(math.Tan),
	"pow":   f2(math.Pow),
	"min":   f2(math.Min),
	"max":   f2(math.Max),
}

// f1 wraps a func(float64) float64 so it accepts any numeric argument.
func f1(fn func(float64) float64) func(any) float64 {
	return func(x any) float64 { return fn(anyToF64(x)) }
}

// f2 wraps a func(float64, float64) float64 so it accepts any numeric arguments.
func f2(fn func(float64, float64) float64) func(any, any) float64 {
	return func(a, b any) float64 { return fn(anyToF64(a), anyToF64(b)) }
}

// anyToF64 converts any numeric value to float64.
func anyToF64(v any) float64 {
	switch n := v.(type) {
	case float64:
		return n
	case float32:
		return float64(n)
	case int:
		return float64(n)
	case int8:
		return float64(n)
	case int16:
		return float64(n)
	case int32:
		return float64(n)
	case int64:
		return float64(n)
	case uint:
		return float64(n)
	case uint64:
		return float64(n)
	}
	return 0
}

// New creates a new CalculatorFunction with the given config.
func New(cfg function.Config) function.AIFunction {
	return &CalculatorFunction{
		Base: function.NewBase(cfg),
	}
}

// Execute implements AIFunction.
func (f *CalculatorFunction) Execute(ctx context.Context, input interface{}) (interface{}, error) {
	var in *Input

	switch v := input.(type) {
	case *Input:
		in = v
	case map[string]interface{}:
		in = &Input{}
		if exprStr, ok := v["expression"].(string); ok {
			in.Expression = exprStr
		}
	default:
		return nil, function.NewExecutionError(f, "input validation", "invalid input type")
	}

	if err := validation.Validate(in); err != nil {
		return nil, function.WrapValidationError(f, err)
	}

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	raw, err := expr.Eval(in.Expression, mathEnv)
	if err != nil {
		return nil, function.WrapExecutionError(f, "evaluate", err)
	}

	var result float64
	switch v := raw.(type) {
	case float64:
		result = v
	case float32:
		result = float64(v)
	case int:
		result = float64(v)
	case int64:
		result = float64(v)
	case uint:
		result = float64(v)
	case uint64:
		result = float64(v)
	default:
		return nil, function.NewExecutionError(f, "result", fmt.Sprintf("expression returned non-numeric type %T", raw))
	}

	// Format as integer string when the result is a whole number.
	var resultStr string
	if result == math.Trunc(result) && !math.IsInf(result, 0) {
		resultStr = fmt.Sprintf("%d", int64(result))
	} else {
		resultStr = fmt.Sprintf("%g", result)
	}

	return &Output{
		Expression: in.Expression,
		Result:     result,
		ResultStr:  resultStr,
	}, nil
}

// InputSchema implements AIFunction.
func (f *CalculatorFunction) InputSchema() *function.Schema {
	return schema.MustGenerateSchema(reflect.TypeOf(Input{}))
}

// Metadata implements AIFunction.
func (f *CalculatorFunction) Metadata() *function.Metadata {
	return function.NewMetadata("CalculatorFunction").
		WithName("calculator").
		WithDescription("Evaluate mathematical expressions").
		WithToolName("calculator").
		WithToolDescription("Evaluates mathematical expressions accurately. Supports arithmetic operators (+, -, *, /), exponentiation via pow(base, exp), and functions: sqrt, abs, floor, ceil, round, log (base10), ln, log2, sin, cos, tan, min, max. Constants: pi, e.").
		WithCLIName("calculate").
		WithCLIDescription("Evaluate a mathematical expression").
		WithCategory("utility").
		WithTags("math", "arithmetic", "calculator").
		WithExamples(
			function.Example{
				Input:       map[string]interface{}{"expression": "sqrt(144) + pow(2, 10)"},
				Output:      map[string]interface{}{"result": 1036, "result_string": "1036"},
				Description: "Square root plus power",
			},
			function.Example{
				Input:       map[string]interface{}{"expression": "355 / 113"},
				Output:      map[string]interface{}{"result_string": "3.1415929203539825"},
				Description: "Rational approximation of pi",
			},
		)
}
