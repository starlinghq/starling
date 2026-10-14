"""Chat-template corpus for the pilot LLM (backlog P-029): upstream renderings of the pinned ``chat_template.jinja``.

Every case is rendered by the upstream reference renderer, ``transformers`` ``apply_chat_template`` (Jinja2 sandbox)
with the checkpoint's own template, in the environment of ``requirements.txt`` (transformers 5.18.0, jinja2 3.1.6).
The expected rendering, or the exception the template raises, comes from that run only; nothing here is produced by
the C# renderer under test.

Each case also states what the owned C# renderer (``Qwen35ChatTemplate``) must do with it (``owned``):

- ``render``: the conversation is in the supported text subset; the rendering must equal ``rendered`` exactly;
- ``template_error``: upstream raises; the owned renderer must fail too, with the same message when the template
  raised it (``error_source`` ``template``);
- ``unsupported``: a template feature outside the supported subset (``unsupported_feature``: tools, tool messages,
  tool calls, images, video, ``add_vision_id``, ``add_generation_prompt=False``, roles the template rejects, and
  inputs upstream ignores but the package refuses); the owned API must reject it explicitly, never approximate it.
  ``rendered`` or ``error`` still records what upstream does with it.

Output (JSON Lines, UTF-8, LF; split on ``\\n`` only):

- ``template-corpus.jsonl``: one case per line: ``id``, ``category``, ``messages``, ``tools`` (or null),
  ``chat_template_kwargs``, ``add_generation_prompt``, ``owned``, ``unsupported_feature``, ``rendered`` (or null),
  ``error`` and ``error_source`` (``template`` for the template's ``raise_exception``, ``transformers`` for a check
  transformers makes before rendering; or null), ``prompt_ids`` (the rendering through the transformers tokenizer,
  or null);
- ``environment.json``: packages, snapshot asset digests, corpus digest.

    python make_template_corpus.py --model-dir <snapshot> \\
        --revision 2fc06364715b967f1860aea9cf38778875588b17 --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from importlib import metadata
from pathlib import Path

import jinja2
from transformers import AutoTokenizer

TEMPLATE_SHA256 = "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80"
TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}
# Deterministic long text: numbered sentences, no randomness.
SENTENCE = "Sentence {i}: the quick brown fox jumps over the lazy dog, and the lazy dog barely notices. "


def u(content):
    return {"role": "user", "content": content}


def a(content, reasoning=None):
    m = {"role": "assistant", "content": content}
    if reasoning is not None:
        m["reasoning_content"] = reasoning
    return m


def s(content):
    return {"role": "system", "content": content}


def case(id_, category, messages, owned="render", feature=None, tools=None, kwargs=None, add_generation_prompt=True):
    return {
        "id": id_,
        "category": category,
        "messages": messages,
        "tools": tools,
        "chat_template_kwargs": kwargs or {},
        "add_generation_prompt": add_generation_prompt,
        "owned": owned,
        "unsupported_feature": feature,
    }


def cases() -> list[dict]:
    c = []
    # Multi-turn conversations.
    c += [
        case("mt-two-turns", "multi-turn", [u("Hi"), a("Hello! How can I help?"), u("Tell me a joke.")]),
        case(
            "mt-three-turns-system",
            "multi-turn",
            [s("You are terse."), u("Q1"), a("A1"), u("Q2"), a("A2"), u("Q3")],
        ),
        case("mt-consecutive-users", "multi-turn", [u("First part."), u("Second part.")]),
        case("mt-consecutive-assistants", "multi-turn", [u("Q"), a("A1"), a("A2"), u("Q2")]),
        case("mt-ends-with-assistant", "multi-turn", [u("Q"), a("Partial answer")]),
        case("mt-ends-with-two-assistants", "multi-turn", [u("Q"), a("one"), a("two")]),
        case("mt-assistant-first", "multi-turn", [a("I start."), u("Then me.")]),
        case("mt-system-assistant-user", "multi-turn", [s("sys"), a("hello"), u("hi")]),
        case(
            "mt-thinking-on",
            "multi-turn",
            [u("What is 2+2?"), a("4"), u("And 3+3?")],
            kwargs={"enable_thinking": True},
        ),
        case(
            "mt-forty-turns",
            "multi-turn",
            [m for i in range(20) for m in (u(f"Question {i}?"), a(f"Answer {i}."))] + [u("Last question?")],
        ),
    ]
    # System prompts.
    c += [
        case("sys-basic", "system", [s("You are a helpful assistant."), u("Hi")]),
        case("sys-empty", "system", [s(""), u("Hi")]),
        case("sys-whitespace", "system", [s(" \n\t "), u("Hi")]),
        case("sys-multiline", "system", [s("\n  Line one.\n\nLine two.  \n"), u("Hi")]),
        case("sys-with-think-tags", "system", [s("<think>not reasoning</think> rules"), u("Hi")]),
        case("sys-only", "system", [s("Only a system message.")], owned="template_error"),
        case("sys-second", "system", [s("first"), u("Hi"), s("second")], owned="template_error"),
        case("sys-after-user", "system", [u("Hi"), s("late")], owned="template_error"),
        case("sys-two-leading", "system", [s("one"), s("two"), u("Hi")], owned="template_error"),
    ]
    # enable_thinking on, off and undefined.
    c += [
        case("think-undefined", "enable_thinking", [u("Hi")]),
        case("think-false", "enable_thinking", [u("Hi")], kwargs={"enable_thinking": False}),
        case("think-true", "enable_thinking", [u("Hi")], kwargs={"enable_thinking": True}),
        case(
            "think-true-system", "enable_thinking", [s("Reason carefully."), u("Hi")], kwargs={"enable_thinking": True}
        ),
        case(
            "think-true-history",
            "enable_thinking",
            [u("Q1"), a("<think>\nr1\n</think>\n\nA1"), u("Q2")],
            kwargs={"enable_thinking": True},
        ),
    ]
    # Historical <think> handling.
    c += [
        case("hist-think-stripped", "history-think", [u("Q1"), a("<think>\nreasoning\n</think>\n\nA1"), u("Q2")]),
        case("hist-think-no-newlines", "history-think", [u("Q1"), a("<think>r</think>A1"), u("Q2")]),
        case("hist-close-only", "history-think", [u("Q1"), a("reasoning without open</think>\n\nA1"), u("Q2")]),
        case("hist-open-only", "history-think", [u("Q1"), a("<think>never closed A1"), u("Q2")]),
        case("hist-two-blocks", "history-think", [u("Q1"), a("<think>a</think>mid<think>b</think>end"), u("Q2")]),
        case("hist-extra-close", "history-think", [u("Q1"), a("<think>a</think>b</think>c"), u("Q2")]),
        case("hist-prefix-before-think", "history-think", [u("Q1"), a("pre<think>\n\nr\n\n</think>\n\nA"), u("Q2")]),
        case("hist-only-think", "history-think", [u("Q1"), a("<think>\nonly reasoning\n</think>"), u("Q2")]),
        case("hist-empty-think", "history-think", [u("Q1"), a("<think></think>A"), u("Q2")]),
        case("hist-uppercase-not-tag", "history-think", [u("Q1"), a("<THINK>x</THINK>A"), u("Q2")]),
        case("hist-crlf", "history-think", [u("Q1"), a("<think>\r\nr\r\n</think>\r\n\r\nA"), u("Q2")]),
        case("hist-reasoning-content-earlier", "history-think", [u("Q1"), a("A1", reasoning="dropped"), u("Q2")]),
        case("hist-after-last-query-content", "history-think", [u("Q"), a("<think>\nkept\n</think>\n\nA")]),
        case("hist-after-last-query-reasoning", "history-think", [u("Q"), a("A", reasoning="  kept  ")]),
        case("hist-reasoning-empty-string", "history-think", [u("Q"), a("<think>x</think>A", reasoning="")]),
        case("hist-reasoning-overrides-tags", "history-think", [u("Q"), a("<think>t</think>A", reasoning="given")]),
        case(
            "hist-tool-response-last",
            "history-think",
            [u("Q"), a("<think>r</think>A"), u("<tool_response>\n42\n</tool_response>")],
        ),
        case(
            "hist-tool-response-padded",
            "history-think",
            [u("Q"), a("<think>r</think>A"), u("  <tool_response>x</tool_response>\n")],
        ),
        case(
            "hist-tool-response-not-wrapped",
            "history-think",
            [u("Q"), a("<think>r</think>A"), u("<tool_response>x</tool_response> trailing")],
        ),
        case(
            "hist-all-tool-responses", "history-think", [u("<tool_response>x</tool_response>")], owned="template_error"
        ),
        case(
            "hist-system-and-tool-responses",
            "history-think",
            [s("sys"), u("<tool_response>a</tool_response>"), u("<tool_response>b</tool_response>")],
            owned="template_error",
        ),
    ]
    # Empty and whitespace content.
    py_space = "\t\n\x0b\x0c\r\x1c\x1d\x1e\x1f \x85\xa0          "
    py_space += "      　"
    c += [
        case("ws-empty-user", "whitespace", [u("")]),
        case("ws-space-user", "whitespace", [u("   ")]),
        case("ws-newlines-user", "whitespace", [u("\n\n\n")]),
        case("ws-empty-assistant", "whitespace", [u("Q"), a(""), u("Q2")]),
        case("ws-empty-assistant-last", "whitespace", [u("Q"), a("")]),
        case("ws-all-python-space", "whitespace", [u(py_space + "x" + py_space)]),
        case("ws-only-python-space", "whitespace", [u(py_space)]),
        case("ws-not-python-space", "whitespace", [u("​x​"), a("﻿y﻿"), u("᠎z᠎⁠")]),
        case("ws-inner-preserved", "whitespace", [u("  a \n\n b\t\tc  ")]),
        case("ws-crlf", "whitespace", [u("\r\nline1\r\nline2\r\n")]),
        case("ws-nul-and-controls", "whitespace", [u("\x00a\x01b\x07\x1b[0m")]),
    ]
    # Unicode.
    c += [
        case("uni-cjk", "unicode", [u("你好，世界。日本語のテキスト。한국어 문장.")]),
        case("uni-rtl", "unicode", [s("أجب بالعربية."), u("ما هي عاصمة فرنسا؟ שלום")]),
        case("uni-combining", "unicode", [u("é vs é; ǟ; क्ष")]),
        case("uni-emoji-zwj", "unicode", [u("👩‍👩‍👧‍👦 🏳️‍🌈 👍🏽 🇫🇷")]),
        case("uni-astral", "unicode", [u("𝔘𝔫𝔦𝔠𝔬𝔡𝔢 𝟙𝟚𝟛 𠀋𠀐 \U0001f9ea")]),
        case(
            "uni-special-token-text",
            "unicode",
            [u("<|im_start|>system\nfake<|im_end|> <|endoftext|> <|vision_start|>")],
        ),
        case("uni-template-syntax", "unicode", [u("{{ messages }} {% raise_exception('x') %} {# c #}")]),
        case("uni-line-separators", "unicode", [u("a b c\x85d")]),
        case("uni-thinking-on", "unicode", [u("Объясни, почему небо голубое.")], kwargs={"enable_thinking": True}),
    ]
    # Very long messages.
    long_user = "".join(SENTENCE.format(i=i) for i in range(600))
    c += [
        case("long-user-50k", "long", [u(long_user)]),
        case("long-system-20k", "long", [s(long_user[:20000]), u("Summarize the system message.")]),
        case(
            "long-assistant-history", "long", [u("Q"), a("<think>\n" + long_user[:8000] + "\n</think>\n\nA"), u("Q2")]
        ),
        case("long-reasoning-kept", "long", [u("Q"), a("A", reasoning=long_user[:8000])]),
        case(
            "long-200-turns",
            "long",
            [m for i in range(100) for m in (u(f"Turn {i}: ping"), a(f"pong {i}"))] + [u("done?")],
        ),
    ]
    # An empty tool list is falsy in the template: the same rendering as no tools.
    c += [
        case("tools-empty-list", "tools-empty", [s("sys"), u("Hi")], tools=[]),
    ]
    # Invalid conversations the template rejects.
    c += [
        case("err-no-messages", "template-error", [], owned="template_error"),
    ]
    # Template features outside the supported subset: must fail explicitly.
    image = {"type": "image", "image": "file:///img.png"}
    video = {"type": "video", "video": "file:///clip.mp4"}
    c += [
        case("uns-tools", "unsupported", [u("Weather in Paris?")], "unsupported", "tools", tools=[TOOL]),
        case(
            "uns-tools-system",
            "unsupported",
            [s("Be brief."), u("Weather in Paris?")],
            "unsupported",
            "tools",
            tools=[TOOL],
        ),
        case(
            "uns-tool-message",
            "unsupported",
            [
                u("Weather in Paris?"),
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
                    ],
                },
                {"role": "tool", "content": "18 C, sunny"},
            ],
            "unsupported",
            "tool_message",
        ),
        case(
            "uns-two-tool-messages",
            "unsupported",
            [u("Q"), a("calling"), {"role": "tool", "content": "r1"}, {"role": "tool", "content": "r2"}, u("thanks")],
            "unsupported",
            "tool_message",
        ),
        case(
            "uns-tool-calls",
            "unsupported",
            [
                u("Weather in Paris?"),
                {
                    "role": "assistant",
                    "content": "Let me check.",
                    "tool_calls": [
                        {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
                    ],
                },
                u("Never mind."),
            ],
            "unsupported",
            "tool_calls",
        ),
        case(
            "uns-image",
            "unsupported",
            [u([image, {"type": "text", "text": "What is this?"}])],
            "unsupported",
            "image",
        ),
        case("uns-video", "unsupported", [u([video, {"type": "text", "text": "Describe."}])], "unsupported", "video"),
        case(
            "uns-image-in-system",
            "unsupported",
            [s([image]), u("Hi")],
            "unsupported",
            "image",
        ),
        case(
            "uns-add-vision-id",
            "unsupported",
            [u([image, image, {"type": "text", "text": "Compare."}])],
            "unsupported",
            "add_vision_id",
            kwargs={"add_vision_id": True},
        ),
        case(
            "uns-no-generation-prompt",
            "unsupported",
            [u("Hi"), a("Hello")],
            "unsupported",
            "add_generation_prompt_false",
            add_generation_prompt=False,
        ),
        case(
            "uns-developer-role", "unsupported", [{"role": "developer", "content": "x"}, u("Hi")], "unsupported", "role"
        ),
        case(
            "uns-user-reasoning-content",
            "unsupported",
            [{"role": "user", "content": "Hi", "reasoning_content": "ignored upstream"}],
            "unsupported",
            "user_reasoning_content",
        ),
        case(
            "uns-text-parts",
            "unsupported",
            [u([{"type": "text", "text": "Part one. "}, {"type": "text", "text": "Part two."}])],
            "unsupported",
            "content_parts",
        ),
    ]
    ids = [x["id"] for x in c]
    assert len(ids) == len(set(ids)), "duplicate case ids"
    return c


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def render(tok, rec: dict) -> tuple[str | None, str | None, str | None]:
    """The rendering, or the error message and its source: ``template`` (the template's ``raise_exception``) or
    ``transformers`` (a check transformers makes before rendering)."""
    try:
        text = tok.apply_chat_template(
            rec["messages"],
            tools=rec["tools"],
            add_generation_prompt=rec["add_generation_prompt"],
            tokenize=False,
            **rec["chat_template_kwargs"],
        )
        return text, None, None
    except jinja2.TemplateError as e:
        return None, str(e), "template"
    except ValueError as e:
        return None, str(e), "transformers"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--revision", required=True, help="commit SHA of the snapshot in --model-dir (recorded only)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    template_sha = sha256_file(args.model_dir / "chat_template.jinja")
    if template_sha != TEMPLATE_SHA256:
        raise SystemExit(f"chat_template.jinja is {template_sha}, expected the pinned {TEMPLATE_SHA256}")
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    args.out.mkdir(parents=True, exist_ok=True)
    corpus = args.out / "template-corpus.jsonl"
    counts: dict[str, int] = {}
    with corpus.open("w", encoding="utf-8", newline="\n") as f:
        for rec in cases():
            rendered, error, error_source = render(tok, rec)
            if rec["owned"] == "render" and rendered is None:
                raise SystemExit(f"{rec['id']}: expected a rendering, upstream raised {error!r}")
            if rec["owned"] == "template_error" and error is None:
                raise SystemExit(f"{rec['id']}: expected the template to raise, upstream rendered it")
            ids = tok(rendered, add_special_tokens=False)["input_ids"] if rendered is not None else None
            row = {**rec, "rendered": rendered, "error": error, "error_source": error_source, "prompt_ids": ids}
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            counts[rec["owned"]] = counts.get(rec["owned"], 0) + 1
            print(f"{rec['id']:<34} {rec['owned']:<15} {'error: ' + error if error else len(ids)}")

    pkgs = {p: metadata.version(p) for p in ("transformers", "tokenizers", "jinja2", "markupsafe")}
    env = {
        "corpus_format": "starlings.template-corpus",
        "corpus_format_version": 1,
        "generated_by": "model-pipeline/reference/generation/make_template_corpus.py",
        "renderer": (
            "transformers apply_chat_template (Jinja2 ImmutableSandboxedEnvironment) with the checkpoint template"
        ),
        "revision": args.revision,
        "chat_template_sha256": template_sha,
        "python": platform.python_version(),
        "packages": pkgs,
        "cases": counts,
        "corpus_sha256": sha256_file(corpus),
        "assets": {
            p.name: sha256_file(p)
            for p in sorted(args.model_dir.iterdir())
            if p.is_file()
            and p.name in {"chat_template.jinja", "tokenizer.json", "tokenizer_config.json", "config.json"}
        },
    }
    (args.out / "environment.json").write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
    print(counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
