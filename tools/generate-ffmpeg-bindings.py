#!/usr/bin/env python3
"""Generates the FFmpeg bindings of the EmbeddingGemma 2 video adapters from FFmpeg's own headers.

One adapter package per FFmpeg major (decision D3, docs/plan/embeddinggemma2-follow-up.md): `--major 8` (the default)
writes Starlings.Models.EmbeddingGemma2.Video.FFmpeg, `--major 9` writes Starlings.Models.EmbeddingGemma2.Video.FFmpeg9.
Both adapters bind the same declarations (SPEC below); only the headers, and so the layouts and library majors, differ.
The headers must be of the selected major (their libavcodec major is checked).

The adapter binds only what it calls: the functions, struct fields and constants declared in SPEC below. Nothing about
the C API is written by hand into the C# code; this script asks the C compiler:

1. A type check (compiled, not run, with mismatches as errors): every function is assigned to a pointer of the declared
   prototype, and every field's C type is asserted with _Static_assert. A declaration that differs from the headers
   fails here.
2. A layout probe (compiled and run): sizeof of each struct, offsetof of each declared field, the value of each constant
   and each library major.

From that it writes FFmpegNative.g.cs: explicit-layout structs with only the declared fields at the compiler's offsets
(and the compiler's struct size), the functions as [LibraryImport] declarations, and the constants. Run it against the
headers of the FFmpeg build the adapter targets (pinned below) on the platform the layout is generated for (x86-64 Linux;
the generated file records the target, and the adapter refuses other process architectures).

    tools/generate-ffmpeg-bindings.py --include <ffmpeg>/include [--major 8|9] [--check]

--check regenerates in memory and fails if the committed file differs (CI-style drift check).

FFmpeg 9 headers: `model-pipeline/reference/embeddinggemma2/fetch_eg2_ffmpeg9_headers.sh <dir>` clones FFmpeg at the
pinned n9.0.1 commit and installs its headers (`configure` + `make install-headers`, nothing compiled) into <dir>/include.
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# Per FFmpeg major: the adapter project the bindings go to, their namespace, the libavcodec major the headers must have,
# and the headers the committed bindings were generated from.
TARGETS = {
    8: {
        "out": ROOT
        / "ai-dotnet/src/Starlings.Models.EmbeddingGemma2.Video.FFmpeg/FFmpegNative.g.cs",
        "namespace": "Starlings.Models.EmbeddingGemma2.Video.FFmpeg",
        "avcodec_major": 62,
        # FFmpeg release branch 8.1, as the adapter's tests use.
        "source": (
            "FFmpeg n8.1.3-17-gc389840894 (commit c3898408946f6cbfe8e244372c250366d5639d00), include/ of the BtbN/FFmpeg-Builds "
            "LGPL shared build ffmpeg-n8.1.3-17-gc389840894-linux64-lgpl-shared-8.1.tar.xz "
            "(SHA-256 bf3db18a4a4bdc066f6b74ee5ec3d7b7be7d6738dcfd7162eae3e09cb857c6b2, "
            "model-pipeline/reference/embeddinggemma2/fetch_eg2_ffmpeg.sh)"
        ),
    },
    9: {
        "out": ROOT
        / "ai-dotnet/src/Starlings.Models.EmbeddingGemma2.Video.FFmpeg9/FFmpegNative.g.cs",
        "namespace": "Starlings.Models.EmbeddingGemma2.Video.FFmpeg9",
        "avcodec_major": 63,
        # FFmpeg release 9.0.1, the release the adapter's tests and exactness gate use.
        "source": (
            "FFmpeg n9.0.1 (commit bf1b838f2ab88b4f8fd83443325c782ea0e0f7fa), include/ from the source tree at that commit "
            "(configure + make install-headers, model-pipeline/reference/embeddinggemma2/fetch_eg2_ffmpeg9_headers.sh)"
        ),
    },
}

HEADERS = [
    "libavutil/avutil.h",
    "libavutil/frame.h",
    "libavutil/display.h",
    "libavutil/pixdesc.h",
    "libavutil/opt.h",
    "libavutil/error.h",
    "libavutil/mem.h",
    "libavutil/log.h",
    "libavcodec/avcodec.h",
    "libavformat/avformat.h",
    "libavformat/avio.h",
    "libswscale/swscale.h",
    "libavfilter/avfilter.h",
    "libavfilter/buffersrc.h",
    "libavfilter/buffersink.h",
    "libswresample/version.h",
]

# Library name, the macro holding its major, and the C# library name used by [LibraryImport].
LIBRARIES = [
    ("avutil", "LIBAVUTIL_VERSION_MAJOR"),
    ("avcodec", "LIBAVCODEC_VERSION_MAJOR"),
    ("avformat", "LIBAVFORMAT_VERSION_MAJOR"),
    ("swscale", "LIBSWSCALE_VERSION_MAJOR"),
    ("avfilter", "LIBAVFILTER_VERSION_MAJOR"),
]

# Libraries loaded only because the bound ones need them (libavcodec and libavfilter link libswresample); no function of
# theirs is bound. Loaded first, by full path, so the bound libraries resolve them without RUNPATH.
DEPENDENCIES = [("swresample", "LIBSWRESAMPLE_VERSION_MAJOR")]

# Structs the adapter reads or writes through a pointer: (C type, C# name, [(field, C type, C# type)]).
# C# types: an explicit field type, or "fixed <type> [n]" for a C array.
STRUCTS = [
    ("AVRational", "AVRational", [("num", "int", "int"), ("den", "int", "int")]),
    (
        "AVFormatContext",
        "AVFormatContext",
        [
            ("pb", "AVIOContext *", "AVIOContext*"),
            ("nb_streams", "unsigned int", "uint"),
            ("streams", "AVStream **", "AVStream**"),
            ("flags", "int", "int"),
        ],
    ),
    (
        "AVStream",
        "AVStream",
        [
            ("codecpar", "AVCodecParameters *", "AVCodecParameters*"),
            ("time_base", "AVRational", "AVRational"),
            ("discard", "enum AVDiscard", "int"),
        ],
    ),
    (
        "AVCodecParameters",
        "AVCodecParameters",
        [
            ("codec_id", "enum AVCodecID", "int"),
            ("coded_side_data", "AVPacketSideData *", "AVPacketSideData*"),
            ("nb_coded_side_data", "int", "int"),
            ("format", "int", "int"),
            ("width", "int", "int"),
            ("height", "int", "int"),
        ],
    ),
    (
        "AVPacketSideData",
        "AVPacketSideData",
        [
            ("data", "uint8_t *", "byte*"),
            ("size", "size_t", "nuint"),
            ("type", "enum AVPacketSideDataType", "int"),
        ],
    ),
    (
        "AVPacket",
        "AVPacket",
        [
            ("pts", "int64_t", "long"),
            ("dts", "int64_t", "long"),
            ("stream_index", "int", "int"),
            ("flags", "int", "int"),
            ("duration", "int64_t", "long"),
        ],
    ),
    (
        "AVCodecContext",
        "AVCodecContext",
        [
            ("thread_count", "int", "int"),
            ("active_thread_type", "int", "int"),
            ("has_b_frames", "int", "int"),
            ("pkt_timebase", "AVRational", "AVRational"),
        ],
    ),
    (
        "AVFrame",
        "AVFrame",
        [
            ("data", "uint8_t *[8]", "fixed ulong [8]"),
            ("pts", "int64_t", "long"),
            ("pkt_dts", "int64_t", "long"),
            ("linesize", "int [8]", "fixed int [8]"),
            ("width", "int", "int"),
            ("height", "int", "int"),
            ("format", "int", "int"),
            ("sample_aspect_ratio", "AVRational", "AVRational"),
            ("color_range", "enum AVColorRange", "int"),
            ("colorspace", "enum AVColorSpace", "int"),
        ],
    ),
    ("AVIOContext", "AVIOContext", [("buffer", "unsigned char *", "byte*")]),
    (
        "AVBufferSrcParameters",
        "AVBufferSrcParameters",
        [
            ("format", "int", "int"),
            ("time_base", "AVRational", "AVRational"),
            ("width", "int", "int"),
            ("height", "int", "int"),
            ("sample_aspect_ratio", "AVRational", "AVRational"),
        ],
    ),
    (
        "AVFilterInOut",
        "AVFilterInOut",
        [
            ("name", "char *", "byte*"),
            ("filter_ctx", "AVFilterContext *", "AVFilterContext*"),
            ("pad_idx", "int", "int"),
            ("next", "struct AVFilterInOut *", "AVFilterInOut*"),
        ],
    ),
]

# Types only ever handled through pointers.
OPAQUE = [
    "AVCodec",
    "SwsContext",
    "SwsFilter",
    "AVFilterGraph",
    "AVFilterContext",
    "AVFilter",
    "AVDictionary",
    "AVInputFormat",
]

# Function-pointer parameter types: name, C typedef, C# type.
CALLBACKS = [
    (
        "avio_read_packet_fn",
        "int (*)(void *opaque, uint8_t *buf, int buf_size)",
        "delegate* unmanaged<void*, byte*, int, int>",
    ),
    (
        "avio_write_packet_fn",
        "int (*)(void *opaque, const uint8_t *buf, int buf_size)",
        "delegate* unmanaged<void*, byte*, int, int>",
    ),
    (
        "avio_seek_fn",
        "int64_t (*)(void *opaque, int64_t offset, int whence)",
        "delegate* unmanaged<void*, long, int, long>",
    ),
]

# Functions: library, C prototype exactly as in the headers.
FUNCTIONS = [
    ("avutil", "unsigned avutil_version(void)"),
    ("avutil", "const char *avutil_license(void)"),
    ("avutil", "const char *av_version_info(void)"),
    ("avutil", "void av_log_set_level(int level)"),
    ("avutil", "int av_strerror(int errnum, char *errbuf, size_t errbuf_size)"),
    ("avutil", "void *av_malloc(size_t size)"),
    ("avutil", "void av_free(void *ptr)"),
    ("avutil", "void av_freep(void *ptr)"),
    ("avutil", "char *av_strdup(const char *s)"),
    ("avutil", "double av_display_rotation_get(const int32_t matrix[9])"),
    ("avutil", "const char *av_get_pix_fmt_name(enum AVPixelFormat pix_fmt)"),
    ("avutil", "AVFrame *av_frame_alloc(void)"),
    ("avutil", "void av_frame_free(AVFrame **frame)"),
    ("avutil", "void av_frame_unref(AVFrame *frame)"),
    (
        "avutil",
        "int av_opt_set(void *obj, const char *name, const char *val, int search_flags)",
    ),
    ("avcodec", "unsigned avcodec_version(void)"),
    ("avcodec", "const char *avcodec_license(void)"),
    ("avcodec", "const char *avcodec_configuration(void)"),
    ("avcodec", "AVPacket *av_packet_alloc(void)"),
    ("avcodec", "void av_packet_free(AVPacket **pkt)"),
    ("avcodec", "void av_packet_unref(AVPacket *pkt)"),
    (
        "avcodec",
        "const AVPacketSideData *av_packet_side_data_get(const AVPacketSideData *sd, int nb_sd, enum AVPacketSideDataType type)",
    ),
    ("avcodec", "const char *avcodec_get_name(enum AVCodecID id)"),
    ("avcodec", "const AVCodec *avcodec_find_decoder(enum AVCodecID id)"),
    ("avcodec", "AVCodecContext *avcodec_alloc_context3(const AVCodec *codec)"),
    (
        "avcodec",
        "int avcodec_parameters_to_context(AVCodecContext *codec, const struct AVCodecParameters *par)",
    ),
    (
        "avcodec",
        "int avcodec_open2(AVCodecContext *avctx, const AVCodec *codec, AVDictionary **options)",
    ),
    (
        "avcodec",
        "int avcodec_send_packet(AVCodecContext *avctx, const AVPacket *avpkt)",
    ),
    ("avcodec", "int avcodec_receive_frame(AVCodecContext *avctx, AVFrame *frame)"),
    ("avcodec", "void avcodec_free_context(AVCodecContext **avctx)"),
    ("avcodec", "void avcodec_flush_buffers(AVCodecContext *avctx)"),
    ("avformat", "unsigned avformat_version(void)"),
    ("avformat", "const char *avformat_license(void)"),
    ("avformat", "int av_read_frame(AVFormatContext *s, AVPacket *pkt)"),
    (
        "avformat",
        "int avformat_seek_file(AVFormatContext *s, int stream_index, int64_t min_ts, int64_t ts, int64_t max_ts, int flags)",
    ),
    ("avformat", "AVFormatContext *avformat_alloc_context(void)"),
    (
        "avformat",
        "int avformat_open_input(AVFormatContext **ps, const char *url, const AVInputFormat *fmt, AVDictionary **options)",
    ),
    (
        "avformat",
        "int avformat_find_stream_info(AVFormatContext *ic, AVDictionary **options)",
    ),
    (
        "avformat",
        "int av_find_best_stream(AVFormatContext *ic, enum AVMediaType type, int wanted_stream_nb, int related_stream, const struct AVCodec **decoder_ret, int flags)",
    ),
    ("avformat", "void avformat_free_context(AVFormatContext *s)"),
    ("avformat", "void avformat_close_input(AVFormatContext **s)"),
    (
        "avformat",
        "AVIOContext *avio_alloc_context(unsigned char *buffer, int buffer_size, int write_flag, void *opaque, avio_read_packet_fn read_packet, avio_write_packet_fn write_packet, avio_seek_fn seek)",
    ),
    ("avformat", "void avio_context_free(AVIOContext **s)"),
    ("swscale", "unsigned swscale_version(void)"),
    ("swscale", "const char *swscale_license(void)"),
    ("swscale", "void sws_freeContext(SwsContext *swsContext)"),
    (
        "swscale",
        "SwsContext *sws_getContext(int srcW, int srcH, enum AVPixelFormat srcFormat, int dstW, int dstH, enum AVPixelFormat dstFormat, int flags, SwsFilter *srcFilter, SwsFilter *dstFilter, const double *param)",
    ),
    (
        "swscale",
        "int sws_getColorspaceDetails(SwsContext *c, int **inv_table, int *srcRange, int **table, int *dstRange, int *brightness, int *contrast, int *saturation)",
    ),
    ("swscale", "const int *sws_getCoefficients(int colorspace)"),
    (
        "swscale",
        "int sws_setColorspaceDetails(SwsContext *c, const int inv_table[4], int srcRange, const int table[4], int dstRange, int brightness, int contrast, int saturation)",
    ),
    (
        "swscale",
        "int sws_scale(SwsContext *c, const uint8_t *const srcSlice[], const int srcStride[], int srcSliceY, int srcSliceH, uint8_t *const dst[], const int dstStride[])",
    ),
    ("avfilter", "unsigned avfilter_version(void)"),
    ("avfilter", "const char *avfilter_license(void)"),
    (
        "avfilter",
        "int av_buffersrc_write_frame(AVFilterContext *ctx, const AVFrame *frame)",
    ),
    ("avfilter", "int av_buffersink_get_frame(AVFilterContext *ctx, AVFrame *frame)"),
    ("avfilter", "AVFilterGraph *avfilter_graph_alloc(void)"),
    ("avfilter", "AVBufferSrcParameters *av_buffersrc_parameters_alloc(void)"),
    (
        "avfilter",
        "AVFilterContext *avfilter_graph_alloc_filter(AVFilterGraph *graph, const AVFilter *filter, const char *name)",
    ),
    ("avfilter", "const AVFilter *avfilter_get_by_name(const char *name)"),
    (
        "avfilter",
        "int av_buffersrc_parameters_set(AVFilterContext *ctx, AVBufferSrcParameters *param)",
    ),
    ("avfilter", "int avfilter_init_str(AVFilterContext *ctx, const char *args)"),
    ("avfilter", "AVFilterInOut *avfilter_inout_alloc(void)"),
    (
        "avfilter",
        "int avfilter_graph_parse_ptr(AVFilterGraph *graph, const char *filters, AVFilterInOut **inputs, AVFilterInOut **outputs, void *log_ctx)",
    ),
    ("avfilter", "void avfilter_inout_free(AVFilterInOut **inout)"),
    ("avfilter", "int avfilter_graph_config(AVFilterGraph *graphctx, void *log_ctx)"),
    ("avfilter", "void avfilter_graph_free(AVFilterGraph **graph)"),
]

# Constants: C# name, C expression, C# type. errno-based codes are those of the generating platform (see EAGAIN note).
CONSTANTS = [
    ("AV_LOG_ERROR", "AV_LOG_ERROR", "int"),
    ("AV_PKT_FLAG_DISCARD", "AV_PKT_FLAG_DISCARD", "int"),
    ("AV_PKT_FLAG_KEY", "AV_PKT_FLAG_KEY", "int"),
    ("AVDISCARD_ALL", "AVDISCARD_ALL", "int"),
    ("FF_THREAD_FRAME", "FF_THREAD_FRAME", "int"),
    ("AV_NOPTS_VALUE", "AV_NOPTS_VALUE", "long"),
    ("AVERROR_EOF", "AVERROR_EOF", "int"),
    ("AVERROR_EINVAL", "AVERROR(EINVAL)", "int"),
    ("AVERROR_EIO", "AVERROR(EIO)", "int"),
    ("AVERROR_EAGAIN_LINUX", "AVERROR(EAGAIN)", "int"),
    ("AVFMT_FLAG_CUSTOM_IO", "AVFMT_FLAG_CUSTOM_IO", "int"),
    ("AVSEEK_SIZE", "AVSEEK_SIZE", "int"),
    ("AV_OPT_SEARCH_CHILDREN", "AV_OPT_SEARCH_CHILDREN", "int"),
    ("AVMEDIA_TYPE_VIDEO", "AVMEDIA_TYPE_VIDEO", "int"),
    ("AV_PKT_DATA_DISPLAYMATRIX", "AV_PKT_DATA_DISPLAYMATRIX", "int"),
    ("AV_PIX_FMT_RGB24", "AV_PIX_FMT_RGB24", "int"),
    ("AVCOL_RANGE_UNSPECIFIED", "AVCOL_RANGE_UNSPECIFIED", "int"),
    ("AVCOL_RANGE_JPEG", "AVCOL_RANGE_JPEG", "int"),
]

CS_SCALAR = {
    "void": "void",
    "int": "int",
    "unsigned": "uint",
    "unsigned int": "uint",
    "double": "double",
    "size_t": "nuint",
    "int64_t": "long",
    "int32_t": "int",
    "uint8_t": "byte",
    "char": "byte",
    "unsigned char": "byte",
}
CS_NAMED = {name: name for name in [s[1] for s in STRUCTS] + OPAQUE}
CS_CALLBACK = {name: cs for name, _, cs in CALLBACKS}


def _cs_type(ctype: str) -> str:
    """C type (as written in a prototype) to the C# interop type. Arrays decay to pointers; enums are int."""
    t = ctype.replace("const", " ").replace("struct ", " ").replace("enum ", "enum:")
    stars = t.count("*") + t.count("[")
    base = re.sub(r"\[\d*\]|\*", " ", t).split()
    base = " ".join(base)
    if base in CS_CALLBACK:
        return CS_CALLBACK[base]
    if base.startswith("enum:"):
        core = "int"
    elif base in CS_SCALAR:
        core = CS_SCALAR[base]
    elif base in CS_NAMED:
        core = CS_NAMED[base]
    else:
        raise SystemExit(f"no C# mapping for C type {ctype!r}")
    return core + "*" * stars


def _parse(prototype: str) -> tuple[str, str, list[tuple[str, str]]]:
    m = re.fullmatch(r"\s*(.+?)\s*\b(\w+)\s*\((.*)\)\s*", prototype)
    if not m:
        raise SystemExit(f"cannot parse {prototype!r}")
    ret, name, params = m.group(1), m.group(2), m.group(3).strip()
    out = []
    if params != "void":
        for p in params.split(","):
            p = p.strip()
            pm = re.fullmatch(r"(.*?)(\w+)\s*((?:\[\d*\])*)", p)
            if not pm:
                raise SystemExit(f"cannot parse parameter {p!r} of {name}")
            ptype = (pm.group(1) + pm.group(3)).strip()
            out.append((ptype, pm.group(2)))
    return ret, name, out


def _includes() -> str:
    return (
        "".join(f"#include <{h}>\n" for h in HEADERS)
        + "#include <errno.h>\n#include <stddef.h>\n#include <stdio.h>\n"
    )


def type_check_source() -> str:
    lines = [_includes()]
    for name, typedef, _ in CALLBACKS:
        lines.append(f"typedef {typedef.replace('(*)', f'(*{name})')};")
    for _, proto in FUNCTIONS:
        ret, name, params = _parse(proto)
        plist = (
            ", ".join(
                f"{t} {n}"
                if not t.endswith("]")
                else f"{t.split('[')[0]} {n}[{t.split('[', 1)[1]}"
                for t, n in params
            )
            or "void"
        )
        lines.append(f"static {ret} (*const check_{name})({plist}) = {name};")
    for ctype, _, fields in STRUCTS:
        for field, ftype, _ in fields:
            lines.append(
                f"_Static_assert(__builtin_types_compatible_p(__typeof__(((({ctype} *)0)->{field})), {_c_typename(ftype)}), "
                f'"{ctype}.{field} is not {ftype}");'
            )
    lines.append("int main(void) { return 0; }")
    return "\n".join(lines) + "\n"


def _c_typename(ftype: str) -> str:
    m = re.fullmatch(r"(.*?)\s*\[(\d+)\]", ftype)
    return f"{m.group(1)}[{m.group(2)}]" if m else ftype


def probe_source() -> str:
    lines = [_includes(), "int main(void) {"]
    for name, macro in LIBRARIES + DEPENDENCIES:
        lines.append(f'  printf("major {name} %d\\n", (int){macro});')
    for ctype, _, fields in STRUCTS:
        lines.append(f'  printf("size {ctype} %zu\\n", sizeof({ctype}));')
        for field, _, _ in fields:
            lines.append(
                f'  printf("offset {ctype} {field} %zu\\n", offsetof({ctype}, {field}));'
            )
    for cs, expr, _ in CONSTANTS:
        lines.append(f'  printf("const {cs} %lld\\n", (long long)({expr}));')
    lines.append('  printf("pointer %zu\\n", sizeof(void *));')
    lines.append("  return 0;\n}")
    return "\n".join(lines) + "\n"


def run_probes(include: Path) -> dict:
    cc = os.environ.get("CC", "cc")
    with tempfile.TemporaryDirectory(prefix="ffmpeg-bindings-") as tmp:
        check = Path(tmp) / "check.c"
        check.write_text(type_check_source())
        subprocess.run(
            [
                cc,
                "-std=gnu11",
                "-fsyntax-only",
                "-Werror",
                "-Wincompatible-pointer-types",
                "-Wno-deprecated-declarations",
                f"-I{include}",
                str(check),
            ],
            check=True,
        )
        probe = Path(tmp) / "probe.c"
        probe.write_text(probe_source())
        exe = Path(tmp) / "probe"
        subprocess.run(
            [
                cc,
                "-std=gnu11",
                "-Wno-deprecated-declarations",
                f"-I{include}",
                str(probe),
                "-o",
                str(exe),
            ],
            check=True,
        )
        output = subprocess.run(
            [str(exe)], check=True, capture_output=True, text=True
        ).stdout
    facts: dict = {"major": {}, "size": {}, "offset": {}, "const": {}}
    for line in output.splitlines():
        kind, *rest = line.split()
        if kind == "major":
            facts["major"][rest[0]] = int(rest[1])
        elif kind == "size":
            facts["size"][rest[0]] = int(rest[1])
        elif kind == "offset":
            facts["offset"][(rest[0], rest[1])] = int(rest[2])
        elif kind == "const":
            facts["const"][rest[0]] = int(rest[1])
        elif kind == "pointer":
            facts["pointer"] = int(rest[0])
    return facts


def render(facts: dict, major: int = 8) -> str:
    spec = TARGETS[major]
    target = f"{platform.system().lower()}-{platform.machine().lower()}"
    o = []
    w = o.append
    w("// <auto-generated>")
    w(
        "// Generated by tools/generate-ffmpeg-bindings.py from FFmpeg's headers; do not edit by hand."
    )
    w(f"// Source: {spec['source']}.")
    w(
        f"// Layout target: {target}, {facts['pointer'] * 8}-bit pointers. Struct sizes and field offsets are the C compiler's"
    )
    w(
        "// (sizeof/offsetof); function prototypes and field types were type-checked against the headers."
    )
    w("// </auto-generated>")
    w("")
    w("#pragma warning disable")
    w("#nullable disable")
    w("")
    w("using System.Runtime.InteropServices;")
    w("")
    w(f"namespace {spec['namespace']};")
    w("")
    for name in OPAQUE:
        w(
            f"/// <summary>Opaque FFmpeg type <c>{name}</c> (used through pointers only).</summary>"
        )
        w(f"internal struct {name}")
        w("{")
        w("}")
        w("")
    for ctype, cs, fields in STRUCTS:
        size = facts["size"][ctype]
        if ctype == "AVRational":
            w(
                f"/// <summary><c>{ctype}</c> ({size} bytes): passed and stored by value.</summary>"
            )
        else:
            w(
                f"/// <summary><c>{ctype}</c> ({size} bytes); only the fields the adapter uses, at FFmpeg's offsets.</summary>"
            )
        w(f"[StructLayout(LayoutKind.Explicit, Size = {size})]")
        w(f"internal unsafe struct {cs}")
        w("{")
        for field, _, cstype in fields:
            off = facts["offset"][(ctype, field)]
            if cstype.startswith("fixed "):
                _, elem, count = cstype.split()
                w(f"    [FieldOffset({off})] public fixed {elem} {field}{count};")
            else:
                w(f"    [FieldOffset({off})] public {cstype} {field};")
        w("}")
        w("")
    w("/// <summary>The FFmpeg functions and constants the adapter uses.</summary>")
    w("internal static unsafe partial class FFmpegNative")
    w("{")
    w(
        "    /// <summary>Library majors of the FFmpeg release the bindings were generated from.</summary>"
    )
    for name, _ in LIBRARIES + DEPENDENCIES:
        w(f"    public const int {name.upper()}_MAJOR = {facts['major'][name]};")
    w("")
    for cs, _, cstype in CONSTANTS:
        value = facts["const"][cs]
        suffix = "L" if cstype == "long" else ""
        w(f"    public const {cstype} {cs} = {value}{suffix};")
    w("")
    for lib, proto in FUNCTIONS:
        ret, name, params = _parse(proto)
        cs_ret = _cs_type(ret)
        cs_params = ", ".join(
            f"string {n}"
            if _cs_type(t) == "byte*" and "const char" in t
            else f"{_cs_type(t)} {n}"
            for t, n in params
        )
        w(f"    /// <summary><c>{proto}</c></summary>")
        if any(_cs_type(t) == "byte*" and "const char" in t for t, _ in params):
            w(
                f'    [LibraryImport("{lib}", StringMarshalling = StringMarshalling.Utf8)]'
            )
        else:
            w(f'    [LibraryImport("{lib}")]')
        w(f"    public static partial {cs_ret} {name}({cs_params});")
        w("")
    o[-1:] = []
    w("}")
    return "\n".join(o) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--include",
        type=Path,
        required=True,
        help="FFmpeg include directory (headers of the target build)",
    )
    parser.add_argument(
        "--major",
        type=int,
        choices=sorted(TARGETS),
        default=8,
        help="FFmpeg major, which selects the adapter project (default 8)",
    )
    parser.add_argument(
        "--check", action="store_true", help="fail if the committed file differs"
    )
    args = parser.parse_args()
    spec = TARGETS[args.major]
    out = spec["out"]
    facts = run_probes(args.include)
    if facts.get("pointer") != 8:
        raise SystemExit(
            "generate on a 64-bit platform: the adapter binds 64-bit layouts only"
        )
    if facts["major"]["avcodec"] != spec["avcodec_major"]:
        raise SystemExit(
            f"the headers are libavcodec {facts['major']['avcodec']}; FFmpeg {args.major} is libavcodec "
            f"{spec['avcodec_major']} (pass the FFmpeg {args.major} include directory or another --major)"
        )
    text = render(facts, args.major)
    if args.check:
        if not out.exists() or out.read_text() != text:
            print(
                f"{out.relative_to(ROOT)} differs from the headers; regenerate it",
                file=sys.stderr,
            )
            return 1
        print(f"ok:   {out.relative_to(ROOT)} matches the headers")
        return 0
    out.write_text(text)
    print(
        f"wrote {out.relative_to(ROOT)}: {len(FUNCTIONS)} functions, {len(STRUCTS)} structs, {len(CONSTANTS)} constants"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
