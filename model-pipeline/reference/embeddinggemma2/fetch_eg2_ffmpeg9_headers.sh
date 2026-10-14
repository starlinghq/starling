#!/bin/sh
# Install the FFmpeg 9 headers the FFmpeg 9 adapter's bindings are generated from (tools/generate-ffmpeg-bindings.py
# --major 9): FFmpeg n9.0.1, commit bf1b838f2ab88b4f8fd83443325c782ea0e0f7fa, from the FFmpeg repository.
#
# Only the headers: `configure` (to write the two generated headers, libavutil/avconfig.h and libavutil/ffversion.h)
# and `make install-headers`; nothing is compiled. On 2026-10-13 the result equalled the headers of the Arch Linux
# package ffmpeg 2:9.0.1-4 except one function declaration that package adds to libavformat/avformat.h (no bound
# declaration or layout is affected; the generator's --check passes on both).
#
# Usage: fetch_eg2_ffmpeg9_headers.sh <destination directory>    # headers in <destination>/include
set -eu
dest="$(mkdir -p "${1:?destination directory}" && cd "$1" && pwd)"
commit=bf1b838f2ab88b4f8fd83443325c782ea0e0f7fa
src="$dest/src"
if [ ! -d "$src/.git" ]; then
  git init -q "$src"
  # The tag too, so FFmpeg's version.sh writes "n9.0.1" into ffversion.h; the commit, not the tag, is the pin.
  git -C "$src" fetch -q --depth 1 https://git.ffmpeg.org/ffmpeg.git "refs/tags/n9.0.1:refs/tags/n9.0.1"
  git -C "$src" -c advice.detachedHead=false checkout -q "$commit"
fi
have="$(git -C "$src" rev-parse HEAD)"
if [ "$have" != "$commit" ]; then
  echo "expected FFmpeg commit $commit, have $have" >&2
  exit 1
fi
cd "$src"
./configure --prefix="$dest" --disable-programs --disable-doc --disable-everything --disable-x86asm \
  --enable-shared --disable-static >"$dest/configure.log"
make install-headers >"$dest/install-headers.log"
echo "FFmpeg $commit headers in $dest/include"
