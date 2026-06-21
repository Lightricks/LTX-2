# LTX-2 Makefile — local inference helpers
# ────────────────────────────────────────────
# Override any variable on the command line:
#   make infer PROMPT="a dog flying"

# ── Paths (override via env or CLI) ─────────
CHECKPOINT   ?= /home/rjman/data/Video/LTX23/models/official/fp8/ltx-2.3-22b-distilled-fp8.safetensors
GEMMA_DIR    ?= /home/rjman/data/Video/LTX23/models/gemma-hf
UPSCALER     ?= /home/rjman/data/Video/LTX23/models/official/bf16/ltx-2.3-spatial-upscaler-x2-1.1.safetensors

# ── Generation params ───────────────────────
PROMPT       ?= A cat running through a sunlit room
OUTPUT       ?= /tmp/ltx-distill-demo.mp4
WIDTH        ?= 768
HEIGHT       ?= 512
FRAMES       ?= 121
FPS          ?= 24
SEED         ?= 42

# ── Derived ─────────────────────────────────
PYTHON       := .venv/bin/python

# ═══════════════════════════════════════════════
#  Targets
# ═══════════════════════════════════════════════

.PHONY: infer help

## make infer  —  Quick T2V with distilled FP8 checkpoint (via ltx-pipelines)
infer:
	$(PYTHON) infer_distilled.py \
		--distilled-checkpoint-path $(CHECKPOINT) \
		--gemma-root $(GEMMA_DIR) \
		--spatial-upsampler-path $(UPSCALER) \
		--prompt "$(PROMPT)" \
		--output $(OUTPUT) \
		--width $(WIDTH) \
		--height $(HEIGHT) \
		--num-frames $(FRAMES) \
		--frame-rate $(FPS) \
		--seed $(SEED)
	@echo ""
	@echo "✓ Done → $(OUTPUT)"

## help  —  Show available targets
help:
	@echo "LTX-2 Makefile targets:"
	@echo ""
	@echo "  make infer                          Distilled T2V (FP8, 2-stage)"
	@echo ""
	@echo "Override variables:  CHECKPOINT  GEMMA_DIR  UPSCALER  PROMPT  OUTPUT"
	@echo "                     WIDTH  HEIGHT  FRAMES  FPS  SEED"

.DEFAULT_GOAL := help
