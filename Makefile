UV ?= uv
DATASET ?= data/dataset
EPOCHS ?= 100
BATCH ?= 8
DEVICE ?=
UV_CACHE_DIR ?= $(CURDIR)/temp/uv_cache
export UV_CACHE_DIR
export UV_PYTHON_INSTALL_DIR := $(CURDIR)/.python
export UV_PROJECT_ENVIRONMENT := $(CURDIR)/.venv

.PHONY: run install check prepare export review train evaluate

install:
	$(UV) sync --python 3.12 --all-extras

app:
	$(UV) run --no-sync pen-view --dataset "$(DATASET)"

train: DATASET = data/input
train:
	$(UV) run --no-sync pen-train --dataset "$(DATASET)" --epochs $(EPOCHS) --batch $(BATCH) $(if $(DEVICE),--device "$(DEVICE)")
