BIN_DIR  ?= $(HOME)/.local/bin
GO       := go
CMDS     := $(notdir $(wildcard cmd/*))
BINARIES := $(addprefix $(BIN_DIR)/,$(CMDS))

.PHONY: all build install clean test lint

all: install

build:
	$(GO) build ./...

install: $(BINARIES)

$(BIN_DIR)/%: cmd/%
	$(GO) build -o $@ ./cmd/$*

clean:
	rm -f $(BINARIES)

test:
	$(GO) test ./...

lint:
	$(GO) vet ./...
