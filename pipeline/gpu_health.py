"""
GPU/CUDA context-poisoning detection for the alignment pipeline.

A CUDA "unspecified launch failure" (and its siblings: illegal memory
access, device-side assert, misaligned address, uncorrectable ECC, cuBLAS
execution/internal errors) poisons the CUDA context for the rest of the
process — every subsequent GPU call fails the same way. There is no
in-process repair; PyTorch/CUDA offer no context-reset API. Confirmed
directly 2026-08-12: the overnight verse-only-mode run hit this ~10 minutes
in and silently produced zero-score fallback output for the remaining
4,926/5,040 chapters (97.7%) because each individual failure looked like an
isolated, expected per-verse/per-chunk CTC-infeasible case — nothing
distinguished "this one verse's audio is too short" from "the GPU is now
broken for the rest of this process." A fresh process (Step 3, immediately
after) used the same GPU/driver without issue, confirming this is a
process-scoped context failure, not a hardware fault — the fix is a fresh
CUDA context (a process restart), not a device-level reset.

Detection has two layers, combined for coverage without false positives:
  1. Signature match — these specific messages are always fatal/sticky,
     never worth a per-item fallback. Fires on the FIRST occurrence.
  2. Consecutive-failure counter — backstop for any signature not in the
     list above. A handful of genuinely-infeasible verses/chunks in a row
     is possible in real data (rare); the GPU being broken makes literally
     EVERY subsequent call fail, so a small threshold safely tells them
     apart without waiting long.

Both funnel through mms_align_words.py's _align_waveform(), the single
choke point every forced-alignment call site shares (verse-only per-verse,
oversized-chapter per-chunk, gap-fill/drift-correction re-alignment) — one
detector covers all of them.
"""

from __future__ import annotations

# Substrings (lowercased) that mean the CUDA context is poisoned for the
# rest of this process — never worth a per-call fallback, always worth
# aborting the whole run. Deliberately does NOT include "out of memory":
# that's a distinct, often-recoverable condition already handled elsewhere
# (chunking, empty_cache()), and auto-restarting the whole process on every
# OOM would be wasteful and wrong.
_POISON_SIGNATURES = (
    "unspecified launch failure",
    "illegal memory access",
    "device-side assert",
    "misaligned address",
    "uncorrectable ecc error",
    "cublas_status_execution_failed",
    "cublas_status_internal_error",
)

_CONSECUTIVE_FAILURE_THRESHOLD = 5

_consecutive_failures = 0


class CudaContextPoisonedError(RuntimeError):
    """The CUDA context is unusable for the rest of this process.

    Subclasses RuntimeError (not a fresh hierarchy) so it still satisfies
    any `except RuntimeError` written before this existed — call sites
    that must NOT swallow it into a per-item fallback add
    `except CudaContextPoisonedError: raise` above their existing
    RuntimeError handler.
    """


def wrap_if_poisoned(e: Exception) -> Exception:
    """Classify a RuntimeError raised by a GPU forced-alignment call.

    Returns e unchanged when it's still safe to treat as an isolated,
    plausibly-real per-item failure. Returns a CudaContextPoisonedError
    wrapping the same message when either detection layer trips. Always
    returns something to *raise* — never swallows:

        except RuntimeError as e:
            raise gpu_health.wrap_if_poisoned(e) from e
    """
    global _consecutive_failures
    _consecutive_failures += 1
    msg = str(e).lower()
    if any(sig in msg for sig in _POISON_SIGNATURES):
        return CudaContextPoisonedError(str(e))
    if _consecutive_failures >= _CONSECUTIVE_FAILURE_THRESHOLD:
        return CudaContextPoisonedError(
            f"{_consecutive_failures} consecutive alignment failures "
            f"(latest: {e}) — treating as a poisoned CUDA context"
        )
    return e


def note_alignment_success() -> None:
    """Reset the consecutive-failure counter after a real GPU alignment
    call completes without raising."""
    global _consecutive_failures
    _consecutive_failures = 0
