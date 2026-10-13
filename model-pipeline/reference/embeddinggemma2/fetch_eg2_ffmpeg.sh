#!/bin/sh
# Fetch the pinned FFmpeg shared builds used by the EmbeddingGemma 2 video spike.
#
# BtbN/FFmpeg-Builds autobuild of FFmpeg n8.1.3-17-gc389840894 (release branch 8.1,
# commit c389840894). The GPL build supplies the encoders (libx264, libvpx-vp9) used to
# generate the synthetic videos; the LGPL build supplies the decoder libraries that
# torchcodec (the upstream decoder) and the C# prototype load. Each archive is checked by
# SHA-256. Autobuild releases expire upstream; keep the archives next to the outputs.
#
# Usage: fetch_eg2_ffmpeg.sh <destination directory>
set -eu
dest="${1:?destination directory}"
mkdir -p "$dest"
cd "$dest"
tag=autobuild-2026-10-11-13-04
rev=n8.1.3-17-gc389840894
for variant in gpl lgpl; do
  case "$variant" in
    gpl) want=4f58c09f117e63a286f2cc2da2d2934b1a45ddc5d00ba7d19151268a21cc16ba ;;
    lgpl) want=bf3db18a4a4bdc066f6b74ee5ec3d7b7be7d6738dcfd7162eae3e09cb857c6b2 ;;
  esac
  archive="ffmpeg-$rev-linux64-$variant-shared-8.1.tar.xz"
  [ -f "$archive" ] || curl -fsSL -o "$archive" \
    "https://github.com/BtbN/FFmpeg-Builds/releases/download/$tag/$archive"
  have=$(sha256sum "$archive" | cut -d' ' -f1)
  if [ "$have" != "$want" ]; then
    echo "SHA-256 mismatch for $archive: $have" >&2
    exit 1
  fi
  tar xf "$archive"
  echo "$archive $have"
done
