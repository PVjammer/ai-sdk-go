// Package validation provides input validation for AI functions.
//
// This package integrates go-playground/validator to validate function inputs
// using struct tags before execution.
package validation

import (
	"fmt"
	"strings"

	"github.com/go-playground/validator/v10"
)

// Validator wraps go-playground/validator for input validation.
type Validator struct {
	validate *validator.Validate
}

// New creates a new Validator instance.
func New() *Validator {
	return &Validator{
		validate: validator.New(),
	}
}

// Global validator instance for convenience.
var globalValidator = New()

// Validate validates a struct using go-playground/validator tags.
//
// The function checks the struct's validate tags and returns an error
// if validation fails. The error includes all validation failures.
//
// Example:
//
//	type Input struct {
//	    Text string `validate:"required,min=1"`
//	    Age  int    `validate:"min=0,max=150"`
//	}
//
//	err := validation.Validate(&Input{Text: "", Age: 200})
//	// Returns error: "Text: required; Age: max"
func Validate(input interface{}) error {
	return globalValidator.Validate(input)
}

// Validate validates a struct instance.
func (v *Validator) Validate(input interface{}) error {
	err := v.validate.Struct(input)
	if err == nil {
		return nil
	}

	// Convert validator errors to user-friendly format
	if validationErrs, ok := err.(validator.ValidationErrors); ok {
		return formatValidationErrors(validationErrs)
	}

	return err
}

// ValidateVar validates a single variable.
//
// Example:
//
//	err := validation.ValidateVar("test@email", "email")
//	// Validates that the string is a valid email
func ValidateVar(field interface{}, tag string) error {
	return globalValidator.ValidateVar(field, tag)
}

// ValidateVar validates a single variable.
func (v *Validator) ValidateVar(field interface{}, tag string) error {
	return v.validate.Var(field, tag)
}

// formatValidationErrors converts validator.ValidationErrors to a user-friendly error.
func formatValidationErrors(errs validator.ValidationErrors) error {
	var messages []string

	for _, err := range errs {
		msg := formatFieldError(err)
		messages = append(messages, msg)
	}

	return &ValidationError{
		Message: strings.Join(messages, "; "),
		Errors:  errs,
	}
}

// formatFieldError formats a single field validation error.
func formatFieldError(err validator.FieldError) string {
	field := err.Field()
	tag := err.Tag()

	switch tag {
	case "required":
		return fmt.Sprintf("%s is required", field)
	case "min":
		return fmt.Sprintf("%s must be at least %s", field, err.Param())
	case "max":
		return fmt.Sprintf("%s must be at most %s", field, err.Param())
	case "email":
		return fmt.Sprintf("%s must be a valid email", field)
	case "url":
		return fmt.Sprintf("%s must be a valid URL", field)
	case "oneof":
		return fmt.Sprintf("%s must be one of: %s", field, err.Param())
	case "len":
		return fmt.Sprintf("%s must be exactly %s characters", field, err.Param())
	case "eq":
		return fmt.Sprintf("%s must equal %s", field, err.Param())
	case "ne":
		return fmt.Sprintf("%s must not equal %s", field, err.Param())
	case "gt":
		return fmt.Sprintf("%s must be greater than %s", field, err.Param())
	case "gte":
		return fmt.Sprintf("%s must be greater than or equal to %s", field, err.Param())
	case "lt":
		return fmt.Sprintf("%s must be less than %s", field, err.Param())
	case "lte":
		return fmt.Sprintf("%s must be less than or equal to %s", field, err.Param())
	default:
		return fmt.Sprintf("%s failed validation: %s", field, tag)
	}
}

// ValidationError represents validation errors.
type ValidationError struct {
	Message string
	Errors  validator.ValidationErrors
}

// Error returns the error message.
func (e *ValidationError) Error() string {
	return fmt.Sprintf("validation failed: %s", e.Message)
}

// Fields returns a map of field names to error messages.
func (e *ValidationError) Fields() map[string]string {
	result := make(map[string]string)
	for _, err := range e.Errors {
		result[err.Field()] = formatFieldError(err)
	}
	return result
}

// IsValidationError checks if an error is a ValidationError.
func IsValidationError(err error) bool {
	_, ok := err.(*ValidationError)
	return ok
}
