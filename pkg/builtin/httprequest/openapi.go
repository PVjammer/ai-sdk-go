package httprequest

import (
	"context"
	"fmt"
	"net/url"
	"strings"

	"github.com/getkin/kin-openapi/openapi3"
)

// resolveOperation finds an operation in an OpenAPI spec by operationId and
// returns the fully-resolved URL (server base + path with substituted params)
// and the HTTP method.
//
// Spec loading priority: specPath (local file) → specURL (remote).
// Server URL priority: operation-level → path-level → global → "".
func resolveOperation(specPath, specURL, operationID string, pathParams map[string]string) (resolvedURL, method string, err error) {
	if specPath == "" && specURL == "" {
		return "", "", fmt.Errorf("spec_path or spec_url must be provided with operation_id")
	}

	loader := openapi3.NewLoader()
	loader.IsExternalRefsAllowed = true

	var doc *openapi3.T
	if specPath != "" {
		doc, err = loader.LoadFromFile(specPath)
	} else {
		u, parseErr := url.Parse(specURL)
		if parseErr != nil {
			return "", "", fmt.Errorf("invalid spec_url: %w", parseErr)
		}
		doc, err = loader.LoadFromURI(u)
	}
	if err != nil {
		return "", "", fmt.Errorf("failed to load spec: %w", err)
	}

	// Validate — non-fatal so real-world specs that bend the rules still work.
	_ = doc.Validate(context.Background())

	if doc.Paths == nil {
		return "", "", fmt.Errorf("spec has no paths defined")
	}

	// HTTP methods we scan (PathItem has one field per method).
	httpMethods := []string{"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}

	for pathStr, pathItem := range doc.Paths.Map() {
		ops := map[string]*openapi3.Operation{
			"GET":     pathItem.Get,
			"POST":    pathItem.Post,
			"PUT":     pathItem.Put,
			"PATCH":   pathItem.Patch,
			"DELETE":  pathItem.Delete,
			"HEAD":    pathItem.Head,
			"OPTIONS": pathItem.Options,
		}

		for _, httpMethod := range httpMethods {
			op := ops[httpMethod]
			if op == nil || op.OperationID != operationID {
				continue
			}

			// Resolve server base URL (operation → path → global → "").
			// op.Servers is *openapi3.Servers (pointer to slice type alias).
			serverBase := ""
			switch {
			case op.Servers != nil && len(*op.Servers) > 0:
				serverBase = (*op.Servers)[0].URL
			case len(pathItem.Servers) > 0:
				serverBase = pathItem.Servers[0].URL
			case len(doc.Servers) > 0:
				serverBase = doc.Servers[0].URL
			}

			// Substitute path params (e.g. /pet/{petId} → /pet/123).
			resolvedPath := pathStr
			for key, val := range pathParams {
				resolvedPath = strings.ReplaceAll(resolvedPath, "{"+key+"}", val)
			}

			return strings.TrimRight(serverBase, "/") + resolvedPath, httpMethod, nil
		}
	}

	return "", "", fmt.Errorf("operation %q not found in spec", operationID)
}
