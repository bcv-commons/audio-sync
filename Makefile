PYTHON := .venv/bin/python

.PHONY: all align align-whisper align-mms align-fuse \
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
