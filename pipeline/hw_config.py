"""
Local hardware-tuning config for the alignment pipeline.

Every machine that runs this repo (M1 laptop, shared CUDA box, rented GPU
pod, ...) has different VRAM/compute headroom, so the "right" values for
things like --mms-chunk-minutes or --whisper-cpu differ per box. Rather
than having to remember and re-type those flags on every `make align`
invocation, this machine's tuning lives once in conf/hw.local.json
(gitignored — see conf/hw.local.json.example for the documented template).

CLI flags always win: this module only supplies argparse *defaults*, so
`conf/hw.local.json` sets what happens when a flag is omitted, and any flag
passed explicitly (including the BooleanOptionalAction --no-* form) still
overrides it for that one run.

Usage (in each script's argparse setup):
    from hw_config import load_hw_config
    hw = load_hw_config()
    parser.add_argument("--model", default=hw["whisper_model"] or DEFAULT_MODEL)
    parser.add_argument("--whisper-cpu", action=argparse.BooleanOptionalAction,
                         default=hw["whisper_cpu"])
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

HW_CONFIG_PATH = Path("conf/hw.local.json")

# Keys this module understands. Anything else in hw.local.json (e.g. a
# "_notes" documentation block, see hw.local.json.example) is ignored.
_DEFAULTS: Dict[str, Any] = {
    "whisper_model": None,        # None -> each script's own DEFAULT_MODEL
    "whisper_cpu": False,
    "mms_cpu": False,
    "mms_chunk_minutes": None,    # None -> mms_align_words.py's per-device default
    "mms_device": None,           # None -> auto (cuda > mps > cpu)
    "parallel_workers": 1,        # shard_align.py's default worker count — 1 = sequential,
                                   # identical to running align_pipeline.py directly. Only
                                   # raise this after testing (see mms_chunk_minutes' note
                                   # in hw.local.json.example for why guessing from VRAM
                                   # headroom alone isn't reliable).
}

_cache: Optional[Dict[str, Any]] = None


def load_hw_config() -> Dict[str, Any]:
    """Load hw.local.json (once per process) merged over the built-in defaults.

    Missing file, or a file missing some keys, is normal — falls back to
    _DEFAULTS for whatever isn't specified. A malformed file is reported
    (not silently ignored, so a typo doesn't quietly undo your tuning) but
    still doesn't crash the run.
    """
    global _cache
    if _cache is not None:
        return _cache

    cfg = dict(_DEFAULTS)
    if HW_CONFIG_PATH.exists():
        try:
            with open(HW_CONFIG_PATH) as f:
                data = json.load(f)
            overridden = [k for k in _DEFAULTS if k in data and data[k] != _DEFAULTS[k]]
            cfg.update({k: data[k] for k in _DEFAULTS if k in data})
            if overridden:
                print(f"[hw_config] Loaded {HW_CONFIG_PATH}: {', '.join(overridden)}")
        except (json.JSONDecodeError, OSError) as e:
            print(f"[hw_config] WARNING: failed to parse {HW_CONFIG_PATH}: {e} — using built-in defaults")

    _cache = cfg
    return cfg
