PYTHON := .venv/bin/python
PIPELINE := pipeline
CONF := conf
TOOLS := tools

.PHONY: all align align-obs stage-obs \
        import-contrib prepare-cross-source \
        publish-align publish-align-dry \
        quality-check quality-compare quality-report \
        install check test clean help

help: ## Show available targets
	@echo "audio-sync — Bible audio alignment pipeline"
	@echo ""
	@echo "  Set BATCH_ID to the batch manifest ID before running alignment:"
	@echo "    BATCH_ID=<id> make align"
	@echo ""
	@echo "  Alignment pipeline"
	@echo "  ──────────────────"
	@echo "  make align          Full pipeline (whisper -> mms -> fuse)"
	@echo "  make align-obs      OBS narration batch -- stage new languages, align all"
	@echo "  make stage-obs      Stage/refresh _obs_batches/<iso>.json manifests from Door43"
	@echo ""
	@echo "  Content preparation"
	@echo "  ────────────────────"
	@echo "  make import-contrib       Import contrib/ into downloads/contrib/"
	@echo "  make prepare-cross-source Fetch helloAO text for DBT audio-only filesets"
	@echo ""
	@echo "  Publishing"
	@echo "  ──────────"
	@echo "  make publish-align      Publish timing-data + run manifest to cdn.bibel.wiki/align/"
	@echo "  make publish-align-dry  Dry-run (no writes)"
	@echo ""
	@echo "  Quality tools"
	@echo "  ─────────────"
	@echo "  make quality-check   Check timing for structural issues"
	@echo "  make quality-compare Compare pipeline vs downloaded timecode"
	@echo "  make quality-report  Generate detailed quality report"
	@echo ""
	@echo "  Setup"
	@echo "  ─────"
	@echo "  make install         Install Python dependencies (+ CUDA libs if an NVIDIA GPU is detected)"
	@echo "  make check           Verify installation"
	@echo "  make test            Run the unit tests (no GPU needed)"
	@echo ""
	@echo "  Pass extra args via ARGS, e.g.:"
	@echo "    make align ARGS=\"--iso heb\""
	@echo "    make align ARGS=\"--iso heb --template John --skip-mms --skip-fusion\""

# ---------------------------------------------------------------------------
# Alignment pipeline
# ---------------------------------------------------------------------------

align: ## Full alignment pipeline (whisper → mms → fuse)
	$(PYTHON) $(PIPELINE)/align_pipeline.py $(ARGS)

align-obs: ## OBS narration batch — stage new languages, align all sequentially
	internal-scripts/run-obs-batch.sh

stage-obs: ## Stage/refresh _obs_batches/<iso>.json manifests from Door43
	$(PYTHON) $(PIPELINE)/stage_obs_batches.py $(ARGS)

# ---------------------------------------------------------------------------
# Content preparation
# ---------------------------------------------------------------------------

import-contrib: ## Import contrib/ into downloads/contrib/
	$(PYTHON) $(TOOLS)/import_contrib.py $(ARGS)

prepare-cross-source: ## Fetch helloAO text for DBT audio-only filesets
	$(PYTHON) $(TOOLS)/prepare_cross_source.py $(ARGS)

# ---------------------------------------------------------------------------
# Publishing
# ---------------------------------------------------------------------------

publish-align: ## Publish timing-data + run manifest to cdn.bibel.wiki/align/
	scripts/publish-align.sh

publish-align-dry: ## Dry-run publish-align (no writes)
	DRY_RUN=1 scripts/publish-align.sh

# ---------------------------------------------------------------------------
# Quality tools
# ---------------------------------------------------------------------------

quality-check: ## Check timing for structural issues
	$(PYTHON) $(TOOLS)/check_timing_quality.py $(ARGS)

quality-compare: ## Compare pipeline vs downloaded timecode
	$(PYTHON) $(TOOLS)/compare_timing.py $(ARGS)

quality-report: ## Generate detailed quality report
	$(PYTHON) $(TOOLS)/quality_report.py $(ARGS)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

install: ## Install Python dependencies
	$(PYTHON) -m pip install -r $(CONF)/requirements-whisper.txt
	@if command -v nvidia-smi >/dev/null 2>&1; then \
		echo "NVIDIA GPU detected — installing $(CONF)/requirements-cuda.txt"; \
		$(PYTHON) -m pip install -r $(CONF)/requirements-cuda.txt; \
	else \
		echo "No NVIDIA GPU detected (nvidia-smi not found) — skipping $(CONF)/requirements-cuda.txt"; \
	fi

check: ## Verify installation
	$(PYTHON) --version
	@$(PYTHON) -c "import torch; print('torch:', torch.__version__)" 2>/dev/null || echo "torch: not installed"
	@$(PYTHON) -c "import torchaudio; print('torchaudio:', torchaudio.__version__)" 2>/dev/null || echo "torchaudio: not installed"
	@$(PYTHON) -c "\
import torch; \
has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(); \
device = 'cuda' if torch.cuda.is_available() else 'mps' if has_mps else 'cpu'; \
print('compute device:', device, '(no GPU found — alignment will run on CPU, slower)' if device == 'cpu' else '')" \
2>/dev/null || echo "compute device: unknown (torch not installed)"

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------

test: ## Run the unit tests (pytest, no GPU needed)
	$(PYTHON) -m pytest tests/ -q

clean: ## Remove generated timing output
	rm -rf export/timing-data/ word-timing-data/
