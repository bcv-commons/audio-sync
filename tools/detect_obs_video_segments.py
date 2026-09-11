#!/usr/bin/env python3
"""
Detect OBS story segment boundaries directly from video, for video-only
sources with no native-language text (e.g. OBS-OBS4All's Godwari/gdx) —
see tools/cache_obs_images.py's module docstring for the full background.

Method (rewritten 2026-09-08 after extensive debugging — see session
history / _obs_verify/ for the full trail of what was tried and why it
didn't work): ORB keypoint matching + RANSAC homography inlier count
against each of the story's reference illustrations (tools/
cache_obs_images.py's cached JPEGs), sampled at a fixed rate across the
video.

Earlier attempts (all in this order, all replaced):
  1. Motion-only cut detection (frame-to-frame / far-apart grayscale
     diff), forced to produce exactly N-1 cuts. Worked for hard cuts but
     silently duplicated one image and skipped another whenever a real
     transition's motion signal wasn't among the strongest — confirmed
     directly (gdx story 22: images 3 and 4 both mapped to the same
     picture, image 3 never got its own segment).
  2. Whole-frame template correlation (multi-scale TM_CCOEFF_NORMED)
     against each reference image, combined via dense-grid DP, motion-
     candidate-restricted DP, greedy sequential crossover, and confidence-
     margin + longest-non-decreasing-subsequence filtering. All of these
     produced weak, inconsistent results (as few as 3 of 7 images ever
     confidently identified) — the common failure was the underlying
     signal itself, not the combination logic: whole-frame correlation
     against a small (128x72) grayscale crop just doesn't discriminate
     well between hand-painted watercolor-style illustrations.
  3. ORB+RANSAC (this version): confirmed directly against the hardest
     case found (gdx story 22's 1->2 transition, wrongly diagnosed as an
     unusually slow crossfade under method 2) — inlier counts are ~100-190
     for the correct image and single digits for every wrong one, a huge,
     unambiguous margin, and the transition turned out to be perfectly
     sharp (not slow at all — method 2 just couldn't see it). Whole story
     22 scanned cleanly end to end in ~6s.

Images are *supposed* to appear in the OBS story's fixed sequential order
— but confirmed directly 2026-09-08 (gdx stories 13 and 44) that some
OBS4All videos genuinely don't: images matched with strong, sustained
confidence (scores 39-190, nowhere near noise) appeared in a different
time-order than their reference numbering assumes (story 44: image 7's
run precedes image 6's; story 13: a whole cluster plays 10, 8, 9, 7).
Enforcing strict non-decreasing order — the original design — silently
discarded these as "shouldn't happen," merging two genuinely separate,
correctly-identified images into one detected segment.

So there is deliberately NO ordering assumption here anymore: a sample
only needs to (a) clear the MIN_INLIERS confidence floor and (b) be part
of a sustained run of at least MIN_RUN_SAMPLES consecutive confident
samples for the same image (suppressing single-sample noise) to become a
segment, regardless of what image number it is relative to its
neighbors. This trusts ORB+RANSAC's demonstrated reliability (wrong
images have consistently scored single digits across every story
examined, right images 39-190+) rather than an ordering assumption
that's now disproven for at least part of this corpus.

Output files land at export/timing-data/obs-video/<iso>/<story>_timing.json
— a separate tree from export/timing-data/obs/ (real audio-based OBS
alignment) so the two are never confused; this one has no text-anchored
timing at all yet, just video-detected position markers ("we still need
to keep these positions as the markers" for whenever real text is found).
Each entry also records "images" (canonical 1-based image number per
detected segment, in time order) and "run_bounds" (the actual first/last
confident-sample timestamp of each run) alongside "pos" (the derived
segment-boundary midpoints) — needed by the second pass below to know
*which* image is missing and *where* to hunt for it.

Second pass: gap hunting (added 2026-09-09)
--------------------------------------------
After the first pass above, a handful of stories are left with a
`count_mismatch`: N-1 (or fewer) images confidently found out of N. Since
there's no ordering assumption, "missing" is well-defined regardless of
where in the sequence it falls: it's whichever canonical image number
never appears anywhere in `images`. `find_gaps()` computes that, and
bounds a search window from the flanking runs' own confirmed edges
(`run_bounds`), not the derived midpoint — confirmed directly against the
3 gaps found in gdx (01/img2, 04/img7, 48/img6): the true gap is
substantially wider than the pos-array midpoint alone would suggest.

`hunt_gap()` then re-scans just that window, restricted to just the
missing candidate(s) and their immediate flanking (already-known) images,
using several upgrades over the first pass — each one only affordable
because the search space here is tens of seconds and 1-3 images, not
minutes and up to ~18 images:
  - Sampling at HUNT_SAMPLE_STEP_SEC (0.5s) instead of 1.5s.
  - Frames grabbed at HUNT_FRAME_W x HUNT_FRAME_H (higher resolution)
    with HUNT_ORB_FEATURES (2500 vs 1000) — more detail to work with,
    aimed at the "genuinely low keypoint count" failure mode (gdx 01
    img2).
  - Matching against the reference image AND a 3x3 grid of overlapping
    half-size crops of it (build_ref_regions), taking the best score
    across all of them — aimed at the other failure mode (gdx 04/48,
    tight Ken-Burns crop on a busy scene): matching the WHOLE reference
    means most of its keypoints fall outside the frame's field of view
    and can never match, capping the achievable inlier count below
    MIN_INLIERS even when the right picture is genuinely on screen.
    Matching a same-sized crop instead means a much larger fraction of
    the crop's keypoints are actually visible in the frame.
  - A second, independent corroborating signal: multi-scale whole-frame
    template correlation (the method that was abandoned in the first
    pass as too weak to use ALONE — see history above — but still useful
    here as a cross-check once the search space is this small).
  - Confidence is margin-based, not just an absolute floor: a sample only
    counts as evidence for a candidate if it beats every flanking
    (already-confirmed) image's score at that same timestamp AND clears
    a relaxed floor (HUNT_MIN_INLIERS=8, well below the first pass's 15)
    — the "test_stable_runs.py" margin idea, adapted from "beats the
    runner-up candidate" to "beats the already-known neighbors", which is
    the actually relevant question here: not "which image is this" but
    "is this really the missing one, or just the neighbor held a little
    longer than estimated".

If hunt_gap() confirms a run, it's merged back into the story's images/
pos/run_bounds (in correct time order) and the story is re-classified;
if not, the unresolved window and the best (sub-threshold) evidence found
anyway is recorded in the output JSON's "gaps" field for manual review,
rather than silently left unexplained.

Usage:
    python tools/detect_obs_video_segments.py --iso gdx --story 01
    python tools/detect_obs_video_segments.py --iso gdx              # every story with a downloaded video
    python tools/detect_obs_video_segments.py --iso gdx --hunt        # also run the second pass on any count_mismatch story
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

VIDEO_DIR = Path("downloads/obs-video")
MANIFEST_DIR = Path("_obs_batches")
REF_DIR = Path("api-cache/obs-images/raw")
OUTPUT_ROOT = Path("export/timing-data/obs-video")

# Confirmed 2026-09-08 (gdx stories 18, 32): a genuinely real, correctly-
# matched segment can be as short as 3.0s — at 3.0s sampling, MIN_RUN_
# SAMPLES=2 can never be satisfied for a segment that short regardless of
# where its samples happen to fall, silently dropping real content. 1.5s
# guarantees at least 2 samples inside any segment >= 3.0s long.
SAMPLE_STEP_SEC = 1.5
FRAME_W, FRAME_H = 256, 144
ORB_FEATURES = 1000
RANSAC_REPROJ_THRESHOLD = 5.0
LOWE_RATIO = 0.75

# A sample only counts as confidently identifying an image when its
# winning score clears this floor — confirmed 2026-09-08 against gdx
# story 22: a correct match scores ~50-190 inliers; near a real
# transition, a weak/ambiguous frame can still "win" among candidates
# with a single-digit score (e.g. 6), which should NOT be trusted to
# place a boundary. This is the "protect against using the transition
# itself as a comparison point" safeguard.
MIN_INLIERS = 15

# Minimum plausible segment length — anything shorter is flagged for
# review rather than silently trusted (same spirit as the old tool's
# MIN_PLAUSIBLE_SEGMENT_SEC, kept as a sanity net even though the new
# method hasn't produced any short-segment artifacts in testing so far).
MIN_PLAUSIBLE_SEGMENT_SEC = 8.0

# A confirmed run needs at least this many CONSECUTIVE confident samples
# of the same image (suppresses a single-sample noise spike from becoming
# its own spurious segment) — see module docstring on why there's no
# ordering requirement alongside this.
MIN_RUN_SAMPLES = 2

# --- Second pass (gap hunting) tuning — see module docstring section
# "Second pass: gap hunting" for the full rationale behind each of these.
HUNT_SAMPLE_STEP_SEC = 0.5
HUNT_FRAME_W, HUNT_FRAME_H = 480, 270
HUNT_ORB_FEATURES = 2500
HUNT_MIN_INLIERS = 8
HUNT_MIN_RUN_SAMPLES = 2
HUNT_PAD_SEC = 3.0  # search a bit outside the estimated gap window, since
# its edges are themselves confirmed-sample boundaries, not the true
# transition point.

TEMPLATE_CONTENT_W, TEMPLATE_CONTENT_H = 128, 72
TEMPLATE_SCALES = [1.0, 1.3, 1.6, 2.0]


def log(message: str, level: str = "INFO"):
    from datetime import datetime
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [{level}] {message}")


_orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
_bf = cv2.BFMatcher(cv2.NORM_HAMMING)


def get_video_duration(video_path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


def grab_frame_at(video_path: Path, t: float, w: int = FRAME_W, h: int = FRAME_H) -> np.ndarray | None:
    cmd = ["ffmpeg", "-v", "error", "-ss", str(t), "-i", str(video_path),
           "-vframes", "1", "-vf", f"scale={w}:{h}",
           "-pix_fmt", "gray", "-f", "rawvideo", "pipe:1"]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    if len(proc.stdout) < w * h:
        return None
    return np.frombuffer(proc.stdout, dtype=np.uint8).reshape(h, w)


def orb_features(img: np.ndarray):
    return _orb.detectAndCompute(img, None)


def inlier_score(frame_kp, frame_desc, ref_kp, ref_desc) -> int:
    """RANSAC homography inlier count between frame and reference
    keypoint matches — see module docstring for why this beats both raw
    match count and whole-frame correlation."""
    if frame_desc is None or ref_desc is None or len(frame_kp) < 4 or len(ref_kp) < 4:
        return 0
    matches = _bf.knnMatch(frame_desc, ref_desc, k=2)
    good = [m for m, n in matches if m.distance < LOWE_RATIO * n.distance] if matches and len(matches[0]) == 2 else []
    if len(good) < 4:
        return len(good)
    src = np.float32([frame_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([ref_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_REPROJ_THRESHOLD)
    return int(mask.sum()) if mask is not None else len(good)


def merge_adjacent_same_image(runs: list[tuple[int, float, float]]) -> list[tuple[int, float, float]]:
    """Merge consecutive entries in the final run sequence that share the
    same image index. Confirmed 2026-09-10 (rnp story 01, und-x-choudhancy
    story 01): a single real segment can fragment into two-or-more
    adjacent runs when a brief noise blip mid-segment drags the winning
    score below MIN_INLIERS for a sample or two — too short to form its
    own run (MIN_RUN_SAMPLES), but enough to break the confident-sample
    adjacency check that builds `runs`. If nothing else was ever confirmed
    between two same-image runs, they're almost certainly one continuous
    display, not two separate sightings — safe to merge unconditionally,
    unlike the non-adjacent case below."""
    if not runs:
        return runs
    merged = [runs[0]]
    for img, start, end in runs[1:]:
        prev_img, prev_start, prev_end = merged[-1]
        if img == prev_img:
            merged[-1] = (prev_img, prev_start, end)
        else:
            merged.append((img, start, end))
    return merged


def resolve_duplicate_images(runs: list[tuple[int, float, float]]) -> tuple[list[tuple[int, float, float]], list[dict]]:
    """After adjacent-merge, the same image can still appear in two or
    more genuinely separate (non-adjacent) runs — confirmed 2026-09-10
    (und-x-choudhancy story 01): a 2.5s false-positive blip of image 2
    right at the video's start, well before image 2's real ~12s run.
    Keep only the longest/most confident run per duplicated image
    (a momentary false match is far more likely to be short than a real
    display), and report what got dropped rather than silently discarding
    it — same transparency principle as the gap-hunting "gaps" field."""
    from collections import defaultdict
    by_image: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    for r in runs:
        by_image[r[0]].append(r)
    kept: list[tuple[int, float, float]] = []
    dropped: list[dict] = []
    for img, group in by_image.items():
        if len(group) == 1:
            kept.append(group[0])
            continue
        group_sorted = sorted(group, key=lambda r: r[2] - r[1], reverse=True)
        kept.append(group_sorted[0])
        for d_img, d_start, d_end in group_sorted[1:]:
            dropped.append({"image": d_img, "start": round(d_start, 2), "end": round(d_end, 2),
                             "duration": round(d_end - d_start, 2)})
    kept.sort(key=lambda r: r[1])
    return kept, dropped


def expected_segment_count(iso: str, story_id: str) -> int | None:
    ref_manifest_path = Path("api-cache/obs-images/manifest.json")
    if not ref_manifest_path.exists():
        return None
    ref_manifest = json.loads(ref_manifest_path.read_text())
    return sum(1 for v in ref_manifest.values() if isinstance(v, dict) and v.get("story") == story_id)


# --- Second pass (gap hunting) helpers ---------------------------------

_hunt_orb = cv2.ORB_create(nfeatures=HUNT_ORB_FEATURES)


def hunt_orb_features(img: np.ndarray):
    return _hunt_orb.detectAndCompute(img, None)


def build_ref_regions(ref_gray: np.ndarray) -> list[np.ndarray]:
    """Whole reference image plus a 3x3 grid of overlapping half-size
    crops — gives ORB a fair shot when the video frame only ever shows a
    zoomed-in sub-region of the illustration: matched against the WHOLE
    reference, most of its keypoints fall outside the frame's field of
    view and can never contribute a match, capping the achievable inlier
    count regardless of how correct the match actually is."""
    h, w = ref_gray.shape
    regions = [ref_gray]
    rh, rw = h // 2, w // 2
    if rh < 8 or rw < 8:
        return regions
    for fy in (0.0, 0.5, 1.0):
        for fx in (0.0, 0.5, 1.0):
            y, x = int(fy * (h - rh)), int(fx * (w - rw))
            regions.append(ref_gray[y:y + rh, x:x + rw])
    return regions


def best_multi_region_score(frame_kp, frame_desc, region_feats: list[tuple]) -> int:
    return max((inlier_score(frame_kp, frame_desc, rkp, rdesc) for rkp, rdesc in region_feats), default=0)


def build_template_pyramid(ref_gray: np.ndarray, scales=TEMPLATE_SCALES) -> list[np.ndarray]:
    pyramid = []
    for scale in scales:
        rh, rw = int(ref_gray.shape[0] * scale), int(ref_gray.shape[1] * scale)
        if rh < TEMPLATE_CONTENT_H or rw < TEMPLATE_CONTENT_W:
            continue
        pyramid.append(cv2.resize(ref_gray, (rw, rh), interpolation=cv2.INTER_AREA))
    return pyramid


def template_corr_score(frame_small: np.ndarray, pyramid: list[np.ndarray]) -> float:
    """Multi-scale whole-frame correlation — the method proven too weak
    to use ALONE (see module docstring), kept here only as a second,
    independent corroborating signal once ORB has already narrowed things
    down to 1-3 candidate images over tens of seconds."""
    best = -1.0
    for ref_scaled in pyramid:
        result = cv2.matchTemplate(ref_scaled, frame_small, cv2.TM_CCOEFF_NORMED)
        best = max(best, float(result.max()))
    return best


def find_gaps(images: list[int], run_bounds: list[list[float]], duration: float, n_images: int) -> list[dict]:
    """Which canonical image number(s) never appear in `images` at all,
    grouped into contiguous blocks, each bounded by the flanking runs'
    own confirmed edges (run_bounds) — not the derived pos midpoint,
    which sits inside the true gap rather than at its edges."""
    present = set(images)
    missing = sorted(set(range(1, n_images + 1)) - present)
    if not missing:
        return []
    gaps = []
    i = 0
    while i < len(missing):
        j = i
        while j + 1 < len(missing) and missing[j + 1] == missing[j] + 1:
            j += 1
        block = missing[i:j + 1]
        before_k, after_k = block[0] - 1, block[-1] + 1
        start_t, end_t = 0.0, duration
        if before_k in images:
            start_t = run_bounds[images.index(before_k)][1]
        if after_k in images:
            end_t = run_bounds[images.index(after_k)][0]
        gaps.append({"missing": block, "window": [round(start_t, 2), round(end_t, 2)]})
        i = j + 1
    return gaps


def hunt_gap(iso: str, story_id: str, gap_start: float, gap_end: float,
             candidates: list[int], neighbor_images: list[int] | None = None,
             n_images: int | None = None) -> dict:
    """Targeted second-pass search for one or more specific missing
    picture(s) within a known, bounded time window — see module docstring
    section 'Second pass: gap hunting' for the full method."""
    video_path = VIDEO_DIR / iso / f"{story_id}.mp4"
    if n_images is None:
        n_images = expected_segment_count(iso, story_id)

    t0 = max(0.0, gap_start - HUNT_PAD_SEC)
    t1 = gap_end + HUNT_PAD_SEC

    want = sorted(set(candidates))
    if neighbor_images is None:
        neighbor_images = sorted({k for k in (want[0] - 1, want[-1] + 1) if 1 <= k <= n_images})
    compare_set = sorted(set(want) | set(neighbor_images))

    region_feats, template_pyramids = {}, {}
    for k in compare_set:
        ref = cv2.imread(str(REF_DIR / f"{story_id}_{k:02d}.jpg"), cv2.IMREAD_GRAYSCALE)
        region_feats[k] = [hunt_orb_features(r) for r in build_ref_regions(ref)]
        template_pyramids[k] = build_template_pyramid(ref)

    n_samples = int((t1 - t0) // HUNT_SAMPLE_STEP_SEC) + 1
    samples = []  # (t, {k: {"orb": int, "tmpl": float}})
    for i in range(n_samples):
        t = round(t0 + i * HUNT_SAMPLE_STEP_SEC, 2)
        frame = grab_frame_at(video_path, t, HUNT_FRAME_W, HUNT_FRAME_H)
        if frame is None:
            continue
        fkp, fdesc = hunt_orb_features(frame)
        frame_small = cv2.resize(frame, (TEMPLATE_CONTENT_W, TEMPLATE_CONTENT_H), interpolation=cv2.INTER_AREA)
        row = {}
        for k in compare_set:
            row[k] = {
                "orb": best_multi_region_score(fkp, fdesc, region_feats[k]),
                "tmpl": round(template_corr_score(frame_small, template_pyramids[k]), 3),
            }
        samples.append((t, row))

    votes = []  # (t, best_k, score, tmpl)
    for t, row in samples:
        best_k = max(want, key=lambda k: row[k]["orb"])
        best_s = row[best_k]["orb"]
        neighbor_best = max((row[k]["orb"] for k in neighbor_images), default=0)
        if best_s >= HUNT_MIN_INLIERS and best_s > neighbor_best:
            votes.append((t, best_k, best_s, row[best_k]["tmpl"]))

    runs = []
    i = 0
    while i < len(votes):
        j = i
        while j + 1 < len(votes) and votes[j + 1][1] == votes[i][1]:
            j += 1
        if (j - i + 1) >= HUNT_MIN_RUN_SAMPLES:
            chunk = votes[i:j + 1]
            runs.append({
                "image": votes[i][1], "start": votes[i][0], "end": votes[j][0],
                "samples": len(chunk),
                "max_orb": max(v[2] for v in chunk),
                "max_tmpl": max(v[3] for v in chunk),
            })
        i = j + 1

    found = {r["image"] for r in runs}
    still_missing = [k for k in want if k not in found]
    best_evidence = {}
    for k in still_missing:
        if not samples:
            continue
        best = max(samples, key=lambda s: s[1][k]["orb"])
        best_evidence[k] = {"t": best[0], "orb": best[1][k]["orb"], "tmpl": best[1][k]["tmpl"]}

    return {
        "window": [round(t0, 2), round(t1, 2)], "candidates": want,
        "neighbor_images": neighbor_images, "runs": runs,
        "still_missing": still_missing, "best_evidence": best_evidence,
    }


def apply_hunt_results(images: list[int], run_bounds: list[list[float]], hunt_result: dict):
    """Merge confirmed hunt-phase runs back into the main detected
    sequence, in time order, and rebuild pos the same way process_story
    does."""
    entries = list(zip(images, run_bounds))
    for r in hunt_result["runs"]:
        entries.append((r["image"], [r["start"], r["end"]]))
    entries.sort(key=lambda e: e[1][0])
    images2 = [e[0] for e in entries]
    run_bounds2 = [e[1] for e in entries]
    pos2 = [0.0]
    for i in range(1, len(entries)):
        prev_end = entries[i - 1][1][1]
        cur_start = entries[i][1][0]
        pos2.append(round((prev_end + cur_start) / 2, 2))
    return images2, pos2, run_bounds2


def process_story(iso: str, story_id: str) -> dict:
    video_path = VIDEO_DIR / iso / f"{story_id}.mp4"
    if not video_path.exists():
        log(f"{video_path} not found", "ERROR")
        return {"story": story_id, "status": "no_video"}

    n_images = expected_segment_count(iso, story_id)
    if not n_images:
        log(f"no reference image count known for story {story_id} — run tools/cache_obs_images.py first", "ERROR")
        return {"story": story_id, "status": "no_reference"}

    ref_feats = []
    for i in range(1, n_images + 1):
        ref_path = REF_DIR / f"{story_id}_{i:02d}.jpg"
        if not ref_path.exists():
            log(f"missing reference image {ref_path}", "ERROR")
            return {"story": story_id, "status": "no_reference"}
        ref = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)
        ref_feats.append(orb_features(ref))

    duration = get_video_duration(video_path)
    n_samples = int(duration // SAMPLE_STEP_SEC) + 1

    # scan: per sample, score against every reference image (small N —
    # a story has at most ~18 — so this is cheap; no need to restrict to
    # a nearby window the way the earlier, much slower template-matching
    # attempts needed to).
    winners: list[tuple[float, int, int]] = []  # (t, winning image idx 0-based, score)
    for i in range(n_samples):
        t = i * SAMPLE_STEP_SEC
        frame = grab_frame_at(video_path, t)
        if frame is None:
            continue
        fkp, fdesc = orb_features(frame)
        scores = [inlier_score(fkp, fdesc, rkp, rdesc) for rkp, rdesc in ref_feats]
        best_k = int(np.argmax(scores))
        winners.append((t, best_k, scores[best_k]))

    # confident samples only, grouped into consecutive-same-image runs in
    # TIME order — no ordering assumption (see module docstring). A run
    # must have >= MIN_RUN_SAMPLES consecutive confident samples to count,
    # which suppresses single-sample noise the same way the old ordering
    # check accidentally also did (as a side effect, not its actual job).
    confident = [(t, k, s) for t, k, s in winners if s >= MIN_INLIERS]
    runs: list[tuple[int, float, float]] = []  # (image_idx, start_t, end_t)
    i = 0
    while i < len(confident):
        j = i
        while j + 1 < len(confident) and confident[j + 1][1] == confident[i][1]:
            j += 1
        if (j - i + 1) >= MIN_RUN_SAMPLES:
            runs.append((confident[i][1], confident[i][0], confident[j][0]))
        i = j + 1

    runs = merge_adjacent_same_image(runs)
    runs, duplicates_dropped = resolve_duplicate_images(runs)

    images = [r[0] + 1 for r in runs]
    run_bounds = [[r[1], r[2]] for r in runs]

    pos = [0.0]
    for r in range(1, len(runs)):
        prev_end = runs[r - 1][2]
        cur_start = runs[r][1]
        pos.append(round((prev_end + cur_start) / 2, 2))

    write_story_output(iso, story_id, images, pos, run_bounds, duration, duplicates_dropped=duplicates_dropped)

    return finalize_story_result(iso, story_id, images, pos, run_bounds, duration, n_images)


def write_story_output(iso: str, story_id: str, images: list[int], pos: list[float],
                        run_bounds: list[list[float]], duration: float, gaps: list[dict] | None = None,
                        duplicates_dropped: list[dict] | None = None):
    out_dir = OUTPUT_ROOT / iso
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{story_id}_timing.json"
    out = {
        "id": f"{iso} story {story_id}",
        "pos": [round(t, 2) for t in pos],
        "images": images,
        "run_bounds": [[round(a, 2), round(b, 2)] for a, b in run_bounds],
        "note": "video-frame-detected, no text yet",
    }
    if gaps:
        out["gaps"] = gaps
    if duplicates_dropped:
        out["duplicates_dropped"] = duplicates_dropped
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out_path


def finalize_story_result(iso: str, story_id: str, images: list[int], pos: list[float],
                           run_bounds: list[list[float]], duration: float, n_images: int) -> dict:
    segment_lengths = [pos[i + 1] - pos[i] for i in range(len(pos) - 1)] + [duration - pos[-1]] if pos else [duration]
    if len(pos) != n_images:
        status = "count_mismatch"
    elif min(segment_lengths) < MIN_PLAUSIBLE_SEGMENT_SEC:
        status = "short_segment_warning"
    else:
        status = "ok"
    return {
        "story": story_id,
        "status": status,
        "segments": len(pos),
        "expected": n_images,
        "duration": duration,
        "min_seg": min(segment_lengths),
        "max_seg": max(segment_lengths),
        "images": images,
        "pos": pos,
        "run_bounds": run_bounds,
        "n_images": n_images,
        "out_path": str(OUTPUT_ROOT / iso / f"{story_id}_timing.json"),
    }


def run_hunt_pass(iso: str, story_id: str, r: dict) -> dict:
    """Orchestrates the second pass for one count_mismatch story: find
    every gap, hunt each one, merge whatever's confirmed back in, and
    rewrite the timing file — leaving unresolved gaps recorded rather
    than silently dropped."""
    images, run_bounds, duration, n_images = r["images"], r["run_bounds"], r["duration"], r["n_images"]
    gaps = find_gaps(images, run_bounds, duration, n_images)
    unresolved_gaps = []
    for gap in gaps:
        log(f"  hunting for image(s) {gap['missing']} in window {gap['window'][0]:.1f}-{gap['window'][1]:.1f}s")
        hunt_result = hunt_gap(iso, story_id, gap["window"][0], gap["window"][1], gap["missing"], n_images=n_images)
        if hunt_result["runs"]:
            found_desc = ", ".join(f"image {rr['image']} at {rr['start']:.1f}-{rr['end']:.1f}s "
                                    f"(orb={rr['max_orb']}, tmpl={rr['max_tmpl']:.2f})" for rr in hunt_result["runs"])
            log(f"    FOUND: {found_desc}")
            images, pos, run_bounds = apply_hunt_results(images, run_bounds, hunt_result)
        if hunt_result["still_missing"]:
            log(f"    still missing: {hunt_result['still_missing']} "
                f"(best sub-threshold evidence: {hunt_result['best_evidence']})")
            unresolved_gaps.append({
                "missing": hunt_result["still_missing"],
                "window": gap["window"],
                "best_evidence": hunt_result["best_evidence"],
            })

    # Defensive: a hunt-confirmed run could in principle duplicate an
    # image already present from the first pass — same dedup as
    # process_story, just applied to the merged (first-pass + hunt) set.
    merged_runs = [(img, rb[0], rb[1]) for img, rb in zip(images, run_bounds)]
    merged_runs.sort(key=lambda r: r[1])
    merged_runs = merge_adjacent_same_image(merged_runs)
    merged_runs, duplicates_dropped = resolve_duplicate_images(merged_runs)
    images = [r[0] for r in merged_runs]
    run_bounds = [[r[1], r[2]] for r in merged_runs]

    pos = [0.0]
    for i in range(1, len(run_bounds)):
        pos.append(round((run_bounds[i - 1][1] + run_bounds[i][0]) / 2, 2))

    write_story_output(iso, story_id, images, pos, run_bounds, duration, gaps=unresolved_gaps or None,
                        duplicates_dropped=duplicates_dropped)
    return finalize_story_result(iso, story_id, images, pos, run_bounds, duration, n_images)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iso", required=True)
    parser.add_argument("--story", default=None, help="Story id, e.g. 01 — omit to process every downloaded story")
    parser.add_argument("--hunt", action="store_true",
                         help="Run the second-pass gap hunter on any count_mismatch story")
    args = parser.parse_args()

    if args.story:
        story_ids = [args.story]
    else:
        video_dir = VIDEO_DIR / args.iso
        if not video_dir.is_dir():
            log(f"{video_dir} not found", "ERROR")
            sys.exit(1)
        story_ids = sorted(p.stem for p in video_dir.glob("*.mp4"))
        log(f"{len(story_ids)} downloaded video(s) found for {args.iso}")

    results = []
    for story_id in story_ids:
        log(f"--- {args.iso} story {story_id} ---")
        r = process_story(args.iso, story_id)
        if args.hunt and r["status"] == "count_mismatch":
            r = run_hunt_pass(args.iso, story_id, r)
        results.append(r)
        if r["status"] in ("no_video", "no_reference"):
            continue
        flag = "" if r["status"] == "ok" else f"  <-- {r['status']}!"
        log(f"  {r['segments']} segment(s) (expected {r['expected']}), "
            f"{r['duration']:.1f}s video, seg length {r['min_seg']:.1f}-{r['max_seg']:.1f}s{flag}")

    ok = sum(1 for r in results if r["status"] == "ok")
    mismatched = [r["story"] for r in results if r["status"] == "count_mismatch"]
    short_warn = [r["story"] for r in results if r["status"] == "short_segment_warning"]
    missing = [r["story"] for r in results if r["status"] in ("no_video", "no_reference")]
    log(f"Done — {ok}/{len(results)} clean, "
        f"{len(short_warn)} short-segment warning{': ' + ','.join(short_warn) if short_warn else ''}, "
        f"{len(mismatched)} count mismatch{': ' + ','.join(mismatched) if mismatched else ''}, "
        f"{len(missing)} missing{': ' + ','.join(missing) if missing else ''}")


if __name__ == "__main__":
    main()
