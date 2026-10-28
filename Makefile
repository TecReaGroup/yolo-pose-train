UV ?= uv
DATASET ?= data/dataset
EPOCHS ?= 100
BATCH ?= 8
DEVICE ?=
export UV_CACHE_DIR := $(CURDIR)/temp/uv_cache
export UV_PYTHON_INSTALL_DIR := $(CURDIR)/.python
export UV_PROJECT_ENVIRONMENT := $(CURDIR)/.venv

.PHONY: run app install check install-dataset install-train prepare export review train evaluate

run:
	$(UV) run --no-sync pen-view --dataset "$(DATASET)"

app: run

install:
	$(UV) sync --python 3.12

check:
	$(UV) run --no-sync pen-view --dataset "$(DATASET)" --smoke-test

install-dataset:
	$(UV) sync --python 3.12 --extra dataset

install-train:
	$(UV) sync --python 3.12 --extra train

prepare:
	$(UV) run --no-sync pen-prepare

export:
	$(UV) run --no-sync pen-export

review:
	$(UV) run --no-sync pen-review

train:
	$(UV) run --no-sync pen-train --dataset "$(DATASET)" --epochs $(EPOCHS) --batch $(BATCH) $(if $(DEVICE),--device "$(DEVICE)")

evaluate:
	$(UV) run --no-sync pen-evaluate --dataset "$(DATASET)" $(if $(DEVICE),--device "$(DEVICE)")
