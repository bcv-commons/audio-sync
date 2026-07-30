# Rented-GPU deployment image for the alignment pipeline (Whisper + MMS +
# fusion). Built once on a trusted machine, then run unattended on rented
# GPU compute (Vast.ai/RunPod/etc).
#
# Deliberately excluded from this image:
#   - R2 publish credentials — the pipeline never runs with --publish here.
#     Results are synced back to a trusted machine (scripts/fetch-remote-run.sh)
#     which publishes using its own local .env. See README's "Rented GPU
#     deployment" section for the full security rationale.
#   - Bible audio/text — fetched at runtime from DBT/CDN, same as any other
#     run of this pipeline. Nothing pre-baked, nothing cached in the image.
#   - mlx / mlx-whisper — Apple Silicon only, dead weight in a Linux container.
#
# Whisper + MMS model weights ARE baked in at build time below, so a
# container boots ready to run with no Hugging Face access needed at
# runtime — HF_TOKEN only needs to exist at *build* time, not on the
# rented box, further shrinking what has to travel to untrusted compute.

FROM pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime

WORKDIR /app

# System deps for audio decoding (torchaudio/torchcodec need ffmpeg's
# libraries at runtime, not just at build time).
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python deps. requirements-whisper.txt still lists mlx/mlx-whisper (macOS
# only) — install everything else from it explicitly rather than editing
# the shared requirements file, so this image's dependency list is visible
# here rather than hidden behind a second requirements variant.
#
# torchaudio isn't bundled in the pytorch/pytorch base image (only torch
# is) and needs a CUDA-matched wheel — installed with --no-deps so pip
# doesn't try to re-resolve/reinstall the base image's pinned torch build.
COPY requirements-whisper.txt requirements-cuda.txt ./
RUN pip install --no-cache-dir \
        requests python-dotenv faster-whisper torchcodec uroman \
    && pip install --no-cache-dir --no-deps torchaudio \
        --index-url https://download.pytorch.org/whl/cu126 \
    && pip install --no-cache-dir -r requirements-cuda.txt

# Repo code.
COPY . .

# Bake model weights in at build time (needs network + optionally HF_TOKEN
# as a --build-arg / buildkit secret; never becomes part of the running
# container's environment). Whisper 'small' (not large-v3) — matches
# whisper_transcribe.py's DEFAULT_MODEL_FASTER; see that file's comment
# for why (Whisper is only MMS-FA's validation/fallback layer here, not
# the primary alignment source, so large-v3's extra cost buys little).
# Keep this in sync with DEFAULT_MODEL_FASTER — a container run without
# an explicit --model flag needs the baked model to match the code's
# default, or it'll try to download a different one at runtime.
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}
RUN python -c "\
from faster_whisper import WhisperModel; \
WhisperModel('small', device='cpu', compute_type='int8')" \
    && python -c "\
import torchaudio; \
torchaudio.pipelines.MMS_FA.get_model()" \
    && rm -rf /root/.cache/huggingface/xet/logs
ENV HF_TOKEN=""

# BIBLE_API_KEY is the only secret expected at *runtime* — inject it via
# the platform's env-var mechanism at pod launch, never bake it in here.
# (No ENV line for it — that's deliberate; an unset var at build time
# just means download_language_content.py logs an error until it's
# provided at `docker run`/pod-launch time.)

ENTRYPOINT ["python", "align_pipeline.py"]
CMD ["--help"]
