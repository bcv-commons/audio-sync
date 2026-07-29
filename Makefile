PYTHON := .venv/bin/python

.PHONY: all align align-whisper align-mms align-fuse \
        import-contrib prepare-cross-source \
        publish-align publish-align-dry \
        quality-check quality-compare quality-report \
        install check help

help: ## Show available targets
	@echo "audio-sync — Bible audio alignment pipeline"
	@echo ""
	@echo "  Set BATCH_ID to the batch manifest ID before running alignment:"
	@echo "    BATCH_ID=<id> make align"
	@echo ""
	@echo "  Alignment pipeline"
	@echo "  ──────────────────"
	@echo "  make align          Full pipeline (whisper → mms → fuse)"
	@echo "  make align-whisper  Step 1a: Whisper transcription"
	@echo "  make align-mms      Step 1b: MMS forced alignment"
	@echo "  make align-fuse     Step 2:  Fuse Whisper + MMS into final timing"
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
	@echo "  make install         Install Python dependencies"
	@echo "  make check           Verify installation"
	@echo ""
	@echo "  Pass extra args via ARGS, e.g.:"
	@echo "    make align ARGS=\"--iso heb\""
	@echo "    make align-whisper ARGS=\"--iso heb --template John\""

# ---------------------------------------------------------------------------
# Alignment pipeline
# ---------------------------------------------------------------------------

align: ## Full alignment pipeline (whisper → mms → fuse)
	$(PYTHON) align_pipeline.py $(ARGS)

align-whisper: ## Step 1a: Whisper transcription
	$(PYTHON) whisper_transcribe.py $(ARGS)

align-mms: ## Step 1b: MMS forced alignment
	$(PYTHON) mms_align_words.py $(ARGS)

align-fuse: ## Step 2: Fuse Whisper + MMS into final timing
	$(PYTHON) align_words.py $(ARGS)

# ---------------------------------------------------------------------------
# Content preparation
# ---------------------------------------------------------------------------

import-contrib: ## Import contrib/ into downloads/contrib/
	$(PYTHON) import_contrib.py $(ARGS)

prepare-cross-source: ## Fetch helloAO text for DBT audio-only filesets
	$(PYTHON) prepare_cross_source.py $(ARGS)

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
	$(PYTHON) check_timing_quality.py $(ARGS)

quality-compare: ## Compare pipeline vs downloaded timecode
	$(PYTHON) compare_timing.py $(ARGS)

quality-report: ## Generate detailed quality report
	$(PYTHON) quality_report.py $(ARGS)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

install: ## Install Python dependencies
	pip install -r requirements-whisper.txt

check: ## Verify installation
	$(PYTHON) --version
	@$(PYTHON) -c "import torch; print('torch:', torch.__version__)" 2>/dev/null || echo "torch: not installed"
	@$(PYTHON) -c "import torchaudio; print('torchaudio:', torchaudio.__version__)" 2>/dev/null || echo "torchaudio: not installed"

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------

clean: ## Remove generated timing output
	rm -rf export/timing-data/ word-timing-data/
