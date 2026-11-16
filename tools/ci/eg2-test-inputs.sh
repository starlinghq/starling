#!/usr/bin/env bash
# The test inputs of the EmbeddingGemma 2 image and video tests that are not in Git and can be prepared reproducibly,
# each verified against a SHA-256 in Git. Used by the qualification job `model-tests` and by
# tools/cuda-qualification-local.sh, so the weight-dependent image and decoder tests run instead of skipping.
#
#   tools/ci/eg2-test-inputs.sh <dir>
#
# Prepares, under <dir>:
#   images/          the 29 fixture images (STARLING_EMBEDDINGGEMMA2_IMAGES): make_eg2_images.py (seeded synthetic
#                    images and scikit-image photos pinned by commit and SHA-256), run with numpy and Pillow from the
#                    hash-locked model-pipeline/reference/embeddinggemma2/requirements.txt; every file is then checked
#                    against model-pipeline/protected/embeddinggemma-2/image-fp32-cpu/images.json
#   ffmpeg8/, ffmpeg9/  the pinned FFmpeg 8.1 and 9.0 LGPL shared builds (STARLING_FFMPEG_DIR, STARLING_FFMPEG9_DIR):
#                    the archives tools/release-gates.sh uses, each checked by SHA-256
#   release-videos/  the four P-030 release videos (STARLING_EMBEDDINGGEMMA2_RELEASE_VIDEOS) from Wikimedia Commons:
#                    fetch_eg2_release_media.py with the videos of the protected media.json, each file checked by SHA-1,
#                    size and SHA-256
#   env              `export NAME=value` lines for the variables above (source it)
#
# Not prepared, because no reproducible source exists: the 22 synthetic fixture videos (STARLING_EMBEDDINGGEMMA2_VIDEOS;
# make_eg2_videos.py needs the BtbN FFmpeg GPL build, which upstream deleted and nothing here hosts) and the reference
# tensors (STARLING_EMBEDDINGGEMMA2_IMAGE_TENSORS, STARLING_EMBEDDINGGEMMA2_VIDEO_TENSORS). The tests that need them
# keep skipping with the reason.
#
# A part already prepared (its .complete marker) is kept; the images are verified again on every run. Needs uv, curl,
# sha256sum, tar and network access. Exit non-zero when any part fails.
set -euo pipefail
dir="${1:?usage: eg2-test-inputs.sh <dir>}"
repo="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$dir"
dir="$(cd "$dir" && pwd)"
cd "$repo"
EG2=model-pipeline/reference/embeddinggemma2
P=model-pipeline/protected/embeddinggemma-2
FF_SOURCE=https://huggingface.co/datasets/Starlinghq/ffmpeg-lgpl-rebuilds/resolve/caf396f176296fc12620c18a344a31f628a4613a
FF8_LIB=$dir/ffmpeg8/ffmpeg-n8.1.3-17-gc389840894-starling-linux64-lgpl-shared-8.1/lib
FF9_ARCHIVE=ffmpeg-n9.0.2-28-g669f5ab083-starling-linux64-lgpl-shared-9.0.tar.xz
FF9_SHA256=db79bc561add07fe48cbe013f714f4ceaf12e49553e56aff9775eff088f8b5ed
FF9_LIB=$dir/ffmpeg9/ffmpeg-n9.0.2-28-g669f5ab083-starling-linux64-lgpl-shared-9.0/lib
export UV_NO_PROGRESS=1 HF_HUB_DISABLE_TELEMETRY=1

# numpy and Pillow, with the hashes of the hash-locked reference requirements (both have no dependencies).
venv=$dir/venv
if [[ ! -x $venv/bin/python ]]; then
    awk '/^(numpy|pillow)==/ { p = 1 } /^[^ #]/ && !/^(numpy|pillow)==/ { p = 0 } p && !/^ *#/' \
        "$EG2/requirements.txt" >"$dir/requirements-images.txt"
    uv venv -q --python 3.11 "$venv"
    uv pip install -q --python "$venv/bin/python" --require-hashes --no-deps -r "$dir/requirements-images.txt" \
        --index-url https://pypi.org/simple
fi

echo "eg2-test-inputs: images"
if [[ ! -f $dir/images/.complete ]]; then
    rm -rf "$dir/images"
    "$venv/bin/python" -I "$EG2/make_eg2_images.py" --out "$dir/images"
    touch "$dir/images/.complete"
fi
python3 -I - "$P/image-fp32-cpu/images.json" "$dir/images" <<'PY'
import hashlib, json, sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
bad = []
for image in manifest["images"]:
    path = Path(sys.argv[2]) / image["file"]
    have = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "missing"
    if have != image["sha256"]:
        bad.append(f"{image['file']}: {have}, images.json {image['sha256']}")
if bad:
    sys.exit("image set differs from the protected images.json:\n  " + "\n  ".join(bad))
print(f"{len(manifest['images'])} images equal to the protected images.json")
PY

echo "eg2-test-inputs: FFmpeg 8 and 9 (LGPL)"
if [[ ! -f $dir/ffmpeg8/.complete ]]; then
    rm -rf "$dir/ffmpeg8"
    sh "$EG2/fetch_eg2_ffmpeg.sh" "$dir/ffmpeg8"
    rm -f "$dir"/ffmpeg8/*.tar.xz
    touch "$dir/ffmpeg8/.complete"
fi
if [[ ! -f $dir/ffmpeg9/.complete ]]; then
    rm -rf "$dir/ffmpeg9"
    mkdir -p "$dir/ffmpeg9"
    curl -fsSL --retry 3 -o "$dir/ffmpeg9/$FF9_ARCHIVE" "$FF_SOURCE/$FF9_ARCHIVE"
    (cd "$dir/ffmpeg9" && echo "$FF9_SHA256  $FF9_ARCHIVE" | sha256sum -c - && tar xf "$FF9_ARCHIVE" && rm -f "$FF9_ARCHIVE")
    touch "$dir/ffmpeg9/.complete"
fi
[[ -d $FF8_LIB && -d $FF9_LIB ]] || { echo "FFmpeg lib directories missing: $FF8_LIB $FF9_LIB" >&2; exit 1; }

echo "eg2-test-inputs: release videos (Wikimedia Commons)"
if [[ ! -f $dir/release-videos/.complete ]]; then
    rm -rf "$dir/release-videos"
    python3 -I -c 'import json, sys; m = json.load(open(sys.argv[1], encoding="utf-8")); json.dump({"videos": m["videos"]}, open(sys.argv[2], "w", encoding="utf-8"))' \
        "$P/release/media/media.json" "$dir/release-videos.json"
    "$venv/bin/python" -I "$EG2/fetch_eg2_release_media.py" --media "$dir/release-videos.json" --out "$dir/release-videos"
    touch "$dir/release-videos/.complete"
fi

cat >"$dir/env" <<EOF
export STARLING_EMBEDDINGGEMMA2_IMAGES=$dir/images
export STARLING_FFMPEG_DIR=$FF8_LIB
export STARLING_FFMPEG9_DIR=$FF9_LIB
export STARLING_EMBEDDINGGEMMA2_RELEASE_VIDEOS=$dir/release-videos
EOF
cat "$dir/env"
