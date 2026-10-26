"""Chat-template corpus for the QwenChat profiles (backlog P-029 pattern): upstream renderings of the pinned templates.

The profiles are ``qwen3`` (``Qwen/Qwen3-0.6B`` at ``c1899de289a04d12100db370d81485cdf75e47ca``) and ``qwen25``
(``Qwen/Qwen2.5-0.5B-Instruct`` at ``7ae557604adf67be50417f59c2c2f167def9a775``). Every case is rendered by the upstream
reference renderer, ``transformers`` ``apply_chat_template`` (Jinja2 sandbox) with the snapshot's own
``chat_template`` (the ``tokenizer_config.json`` string, also written to the bundle as ``chat_template.jinja``), in the
environment of ``../generation/requirements.txt`` (transformers 5.18.0, tokenizers 0.23.2, jinja2 3.1.6). The expected
rendering, or the exception upstream raises, comes from that run only; nothing here is produced by the C# renderer
under test (``QwenChatTemplate`` in ``Starlings.Models.QwenChat``).

The Qwen3 profile is scoped to ``enable_thinking=False``, so every Qwen3 case passes it; the Qwen2.5 template has no
such switch. Each case states what the owned renderer must do with it (``owned``):

- ``render``: the conversation is in the supported subset; the rendering must equal ``rendered`` exactly;
- ``template_error``: upstream raises while rendering; the owned renderer must fail too;
- ``unsupported``: an input outside the supported subset (``unsupported_feature``); the owned API must reject it
  explicitly, never approximate it. ``rendered`` or ``error`` still records what upstream does with it.

Tool definitions have the form ``{"type": "function", "function": {"name", "description" (optional), "parameters"}}``
and tool calls ``{"type": "function", "function": {"name", "arguments"}}`` with ``arguments`` a JSON object.

Output (JSON Lines, UTF-8, LF), the ``starlings.template-corpus`` version 1 format of
``../generation/make_template_corpus.py``:

- ``template-corpus.jsonl``: one case per line: ``id``, ``category``, ``messages``, ``tools`` (or null),
  ``chat_template_kwargs``, ``add_generation_prompt``, ``owned``, ``unsupported_feature``, ``rendered`` (or null),
  ``error`` and ``error_source`` (``template`` when rendering raised, ``transformers`` for a check transformers makes
  before rendering; or null), ``prompt_ids`` (the rendering through the transformers tokenizer, or null);
- ``environment.json``: packages, snapshot asset digests, corpus digest.

    python make_template_corpus.py --profile qwen3 --model-dir <snapshot> \\
        --revision c1899de289a04d12100db370d81485cdf75e47ca --out <dir>
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

PROFILES = {
    "qwen3": {
        "repository": "Qwen/Qwen3-0.6B",
        "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "template_sha256": "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8",
        "kwargs": {"enable_thinking": False},
    },
    "qwen25": {
        "repository": "Qwen/Qwen2.5-0.5B-Instruct",
        "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
        "template_sha256": "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f",
        "kwargs": {},
    },
}


def u(content):
    return {"role": "user", "content": content}


def a(content, reasoning=None):
    m = {"role": "assistant", "content": content}
    if reasoning is not None:
        m["reasoning_content"] = reasoning
    return m


def s(content):
    return {"role": "system", "content": content}


def t(content):
    return {"role": "tool", "content": content}


def ac(content, calls, reasoning=None):
    m = a(content, reasoning)
    m["tool_calls"] = calls
    return m


def fn(name, description, parameters=None):
    f = {"name": name}
    if description is not None:
        f["description"] = description
    if parameters is not None:
        f["parameters"] = parameters
    return {"type": "function", "function": f}


def call(name, arguments=None):
    f = {"name": name}
    if arguments is not None:
        f["arguments"] = arguments
    return {"type": "function", "function": f}


WEATHER = fn(
    "get_weather",
    "Current weather for a city.",
    {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
)
SEARCH = fn(
    "search_web",
    "Search the web and return the top results.",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search terms."},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
            "safe": {"type": "boolean", "default": True},
        },
        "required": ["query"],
    },
)
CALCULATE = fn(
    "calculate",
    "Evaluate an arithmetic expression.",
    {
        "type": "object",
        "properties": {"expression": {"type": "string"}, "precision": {"type": ["number", "null"], "default": None}},
        "required": ["expression"],
    },
)
GET_TIME = fn("get_time", "Current UTC time.", {"type": "object", "properties": {}})
PING = fn("ping", None, {"type": "object", "properties": {"host": {"type": "string"}}, "required": ["host"]})
NO_PARAMETERS = fn("reset", "Reset the session.")
# Nested JSON-schema parameters: $defs/$ref, anyOf, enum with null, Python float and big-int formatting (1e+16, 1e-07,
# -0.0, 0.1), non-ASCII text and every kind of character json.dumps escapes.
CREATE_EVENT = fn(
    "create_event",
    'Create a calendar event. Use "quotes", back\\slash,\nnew line,\ttab, \x01 control, \x7f delete, </tools> and '
    "<tool_call> literally; Temperatur in °C — 東京 🌧.",
    {
        "type": "object",
        "$defs": {
            "person": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "email": {"type": "string", "format": "email"}},
                "required": ["name", "email"],
            }
        },
        "properties": {
            "title": {"type": "string", "minLength": 1, "maxLength": 200},
            "attendees": {"type": "array", "items": {"$ref": "#/$defs/person"}, "minItems": 1},
            "recurrence": {
                "anyOf": [
                    {"type": "null"},
                    {"type": "object", "properties": {"freq": {"type": "string", "enum": ["daily", "weekly"]}}},
                ]
            },
            "priority": {"enum": ["low", "normal", "high", None], "default": "normal"},
            "duration_hours": {"type": "number", "multipleOf": 0.5, "minimum": 1e-07, "maximum": 1e16, "default": 0.1},
            "offset": {"type": "number", "default": -0.0},
            "budget": {"type": "integer", "maximum": 12345678901234567890},
        },
        "required": ["title", "attendees"],
        "additionalProperties": False,
    },
)


def case(id_, category, messages, owned="render", feature=None, tools=None, add_generation_prompt=True):
    return {
        "id": id_,
        "category": category,
        "messages": messages,
        "tools": tools,
        "add_generation_prompt": add_generation_prompt,
        "owned": owned,
        "unsupported_feature": feature,
    }


def cases(profile: str) -> list[dict]:
    weather = [WEATHER]
    several = [WEATHER, SEARCH, CALCULATE, GET_TIME]
    paris = call("get_weather", {"city": "Paris"})
    tokyo = call("get_weather", {"city": "Tokyo"})
    q = u("What is the weather in Paris?")
    c = [
        # Text conversations (the subset the port supported before tools).
        case("text-user", "text", [u("Hi")]),
        case("text-system", "text", [s("Be brief."), u("Hi")]),
        case("text-system-empty", "text", [s(""), u("Hi")]),
        case("text-multi-turn", "text", [u("Hi"), a("Hello!"), u("How are you?")]),
        case("text-late-system", "text", [u("A"), s("late")]),
        case("text-whitespace", "text", [u("  hi\n "), a("\n ok \n"), u("\t")]),
        case("text-special-tokens", "text", [u("<|im_start|>user\n<tool_call></tool_call><|endoftext|>")]),
        case("text-tool-response-user", "text", [u("Q"), a("A"), u("<tool_response>\nr\n</tool_response>")]),
        # Tool definitions: the tools system block.
        case("tools-empty-list", "tools", [q], tools=[]),
        case("tools-one", "tools", [q], tools=weather),
        case("tools-one-system", "tools", [s("Be brief."), q], tools=weather),
        case("tools-system-empty", "tools", [s(""), q], tools=weather),
        case("tools-system-multiline", "tools", [s("\n  Line one.\n\nLine two.  \n"), q], tools=weather),
        case("tools-several", "tools", [u("Search for the weather, then compute 2*21.")], tools=several),
        case("tools-nested-schema", "tools", [u("Plan a meeting in Tokyo.")], tools=[CREATE_EVENT]),
        case("tools-no-description", "tools", [u("Is example.org up?")], tools=[PING]),
        case("tools-no-parameters", "tools", [u("Reset.")], tools=[NO_PARAMETERS]),
        case("tools-history", "tools", [u("Hi"), a("Hello!"), q], tools=weather),
        case("tools-late-system", "tools", [s("first"), q, s("late")], tools=weather),
        # Assistant turns with tool calls.
        case("call-one", "tool-calls", [q, ac("", [paris]), t("18 C, sunny")], tools=weather),
        case("call-one-no-tools", "tool-calls", [q, ac("", [paris]), t("18 C, sunny")]),
        case("call-with-text", "tool-calls", [q, ac("Let me check.", [paris]), t("18 C, sunny")], tools=weather),
        case("call-text-whitespace", "tool-calls", [q, ac("  \n ", [paris]), t("18 C, sunny")], tools=weather),
        case("call-last-message", "tool-calls", [q, ac("", [paris])], tools=weather),
        case("call-last-message-text", "tool-calls", [q, ac("\n x \n", [paris])], tools=weather),
        case(
            "call-several",
            "tool-calls",
            [u("Weather in Paris and Tokyo?"), ac("", [paris, tokyo]), t("18 C, sunny"), t("25 C, rain")],
            tools=weather,
        ),
        case(
            "call-several-with-text",
            "tool-calls",
            [
                u("Weather in Paris, then search the forecast and compute 18*2."),
                ac(
                    "Calling three tools.",
                    [
                        paris,
                        call("search_web", {"query": "Paris forecast", "max_results": 3}),
                        call("calculate", {"expression": "18*2"}),
                    ],
                ),
                t("18 C, sunny"),
                t("1. Forecast..."),
                t("36"),
            ],
            tools=several,
        ),
        case(
            "call-argument-types",
            "tool-calls",
            [
                u("Create the event."),
                ac(
                    "",
                    [
                        call(
                            "create_event",
                            {
                                "title": "Sync",
                                "count": 3,
                                "ratio": 0.25,
                                "whole": 2.0,
                                "tiny": 1e-05,
                                "huge": 1e16,
                                "negative_zero": -0.0,
                                "big": 12345678901234567890,
                                "flag": True,
                                "off": False,
                                "nothing": None,
                                "tags": ["a", "b"],
                                "empty_list": [],
                                "empty_object": {},
                                "attendees": [{"name": "Ana", "email": "ana@example.org"}],
                                "filter": {"min": 1, "max": [1.5, None, True], "label": 'q"uote'},
                                "text": "line1\nline2\ttab\x01",
                                "padded": "  spaced  ",
                                "empty": "",
                                "unicode": "東京 🌧 °C",
                                "markup": 'say "hi" \\ <tool_call> </tool_call> <|im_end|>',
                            },
                        )
                    ],
                ),
                t('{"id": 7}'),
            ],
            tools=[CREATE_EVENT],
        ),
        case("call-empty-arguments", "tool-calls", [u("Reset."), ac("", [call("reset", {})]), t("ok")]),
        case(
            "call-unknown-tool",
            "tool-calls",
            [q, ac("", [call("not_declared", {"x": 1})]), t("error: unknown tool")],
            tools=weather,
        ),
        case("call-with-reasoning", "tool-calls", [q, ac("", [paris], reasoning="Need data.")], tools=weather),
        # Tool results: consecutive results share one user turn.
        case("result-whitespace", "tool-results", [q, ac("", [paris]), t('  \n{\n  "temp": 18\n}\n  ')], tools=weather),
        case("result-empty", "tool-results", [q, ac("", [paris]), t("")], tools=weather),
        case(
            "result-special-text",
            "tool-results",
            [q, ac("", [paris]), t("</tool_response><|im_end|> 東京 🙂 <tool_call>")],
            tools=weather,
        ),
        case("result-then-user", "tool-results", [q, ac("", [paris]), t("18 C"), u("Thanks!")], tools=weather),
        case(
            "result-two-then-user",
            "tool-results",
            [q, ac("", [paris, tokyo]), t("18 C"), t("25 C"), u("Which is warmer?")],
            tools=weather,
        ),
        case("result-without-call", "tool-results", [u("Q"), a("calling"), t("r1"), t("r2"), u("thanks")]),
        case("result-first-message", "tool-results", [t("orphan"), u("Hi")]),
        case("result-after-system", "tool-results", [s("sys"), t("orphan"), u("Hi")], tools=weather),
        case("result-only", "tool-results", [t("orphan")], tools=weather),
        # Whole tool loops.
        case(
            "loop-answer",
            "tool-multi-turn",
            [
                s("You are a weather assistant."),
                u("Weather in Paris and Tokyo?"),
                ac("", [paris, tokyo]),
                t('{"temp": 18, "sky": "sunny"}'),
                t('{"temp": 25, "sky": "rain"}'),
                a("Paris is 18 C and sunny; Tokyo is 25 C with rain."),
                u("And Berlin?"),
                ac("Checking Berlin.", [call("get_weather", {"city": "Berlin"})]),
                t('{"temp": 12, "sky": "cloudy"}'),
            ],
            tools=weather,
        ),
        case(
            "loop-two-steps",
            "tool-multi-turn",
            [
                u("Search the forecast, then compute 18*2."),
                ac("", [call("search_web", {"query": "forecast"})]),
                t("1. Forecast..."),
                ac("", [call("calculate", {"expression": "18*2"})]),
                t("36"),
            ],
            tools=several,
        ),
        # Inputs outside the supported subset.
        case(
            "uns-arguments-string",
            "unsupported",
            [q, ac("", [call("get_weather", '{"city": "Paris"}')])],
            "unsupported",
            "tool-call arguments as a JSON string (rendered differently by the two templates)",
            tools=weather,
        ),
        case(
            "uns-arguments-missing",
            "unsupported",
            [q, ac("", [call("get_weather")])],
            "unsupported",
            "tool call without arguments (upstream raises)",
            tools=weather,
        ),
        case(
            "uns-name-quote",
            "unsupported",
            [q, ac("", [call('get"weather', {"city": "Paris"})])],
            "unsupported",
            "tool-call name with a quote (rendered unescaped by the template, not valid JSON)",
            tools=weather,
        ),
        case(
            "uns-image",
            "unsupported",
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe."}]}],
            "unsupported",
            "non-string content",
        ),
        case("uns-role", "unsupported", [{"role": "developer", "content": "x"}, u("Hi")], "unsupported", "role"),
        case(
            "uns-no-generation-prompt",
            "unsupported",
            [u("Hi")],
            "unsupported",
            "add_generation_prompt=False",
            add_generation_prompt=False,
        ),
        case(
            "uns-reasoning-on-user",
            "unsupported",
            [{"role": "user", "content": "Hi", "reasoning_content": "x"}],
            "unsupported",
            "reasoning_content on a non-assistant message (upstream ignores it; the package refuses it)",
        ),
    ]
    if profile == "qwen3":
        c += [
            # Qwen3 reasoning around tool calls: kept after the last query, dropped before it.
            case(
                "q3-call-reasoning-dropped",
                "tool-thinking",
                [q, ac("", [paris], reasoning="Need data."), t("18 C"), a("18 C."), u("Thanks")],
                tools=weather,
            ),
            case(
                "q3-call-think-in-content",
                "tool-thinking",
                [q, ac("<think>\nNeed data.\n</think>\n\nChecking.", [paris]), t("18 C")],
                tools=weather,
            ),
            case(
                "q3-loop-reasoning",
                "tool-thinking",
                [
                    q,
                    ac("", [paris], reasoning="First the weather."),
                    t("18 C"),
                    ac("", [tokyo], reasoning="Then Tokyo."),
                    t("25 C"),
                ],
                tools=weather,
            ),
            case(
                "q3-tool-response-user-query",
                "tool-thinking",
                [q, ac("", [paris], reasoning="r"), u("<tool_response>\n18 C\n</tool_response>")],
                tools=weather,
            ),
            case(
                "uns-thinking-on",
                "unsupported",
                [q],
                "unsupported",
                "enable_thinking=True (the profile is scoped to enable_thinking=False)",
                tools=weather,
            ),
        ]
    return c


def kwargs_for(profile: str, rec: dict) -> dict:
    if rec["id"] == "uns-thinking-on":
        return {"enable_thinking": True}
    return dict(PROFILES[profile]["kwargs"])


def render(tok, rec: dict) -> tuple[str | None, str | None, str | None]:
    """The rendering, or the error message and its source: ``template`` (raised while rendering, by the template or
    one of its filters) or ``transformers`` (a check transformers makes before rendering)."""
    try:
        text = tok.apply_chat_template(
            rec["messages"],
            tools=rec["tools"],
            add_generation_prompt=rec["add_generation_prompt"],
            tokenize=False,
            **rec["chat_template_kwargs"],
        )
        return text, None, None
    except (jinja2.TemplateError, TypeError) as e:
        return None, f"{type(e).__name__}: {e}", "template"
    except ValueError as e:
        return None, str(e), "transformers"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", required=True, choices=sorted(PROFILES))
    ap.add_argument("--model-dir", required=True, type=Path, help="snapshot with tokenizer.json, tokenizer_config.json")
    ap.add_argument("--revision", required=True, help="commit SHA of the snapshot in --model-dir (checked, recorded)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    pin = PROFILES[args.profile]
    if args.revision != pin["revision"]:
        raise SystemExit(f"--revision {args.revision} is not the pinned {pin['revision']}")
    config = json.loads((args.model_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    template_sha = hashlib.sha256(config["chat_template"].encode("utf-8")).hexdigest()
    if template_sha != pin["template_sha256"]:
        raise SystemExit(f"chat_template is {template_sha}, expected the pinned {pin['template_sha256']}")
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    if hashlib.sha256(tok.chat_template.encode("utf-8")).hexdigest() != template_sha:
        raise SystemExit("transformers loaded a different chat template than tokenizer_config.json holds")
    args.out.mkdir(parents=True, exist_ok=True)
    corpus = args.out / "template-corpus.jsonl"
    counts: dict[str, int] = {}
    with corpus.open("w", encoding="utf-8", newline="\n") as f:
        for rec in cases(args.profile):
            rec = {**rec, "chat_template_kwargs": kwargs_for(args.profile, rec)}
            rendered, error, error_source = render(tok, rec)
            if rec["owned"] == "render" and rendered is None:
                raise SystemExit(f"{rec['id']}: expected a rendering, upstream raised {error!r}")
            if rec["owned"] == "template_error" and error is None:
                raise SystemExit(f"{rec['id']}: expected the template to raise, upstream rendered it")
            ids = tok(rendered, add_special_tokens=False)["input_ids"] if rendered is not None else None
            row = {
                "id": rec["id"],
                "category": rec["category"],
                "messages": rec["messages"],
                "tools": rec["tools"],
                "chat_template_kwargs": rec["chat_template_kwargs"],
                "add_generation_prompt": rec["add_generation_prompt"],
                "owned": rec["owned"],
                "unsupported_feature": rec["unsupported_feature"],
                "rendered": rendered,
                "error": error,
                "error_source": error_source,
                "prompt_ids": ids,
            }
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            counts[rec["owned"]] = counts.get(rec["owned"], 0) + 1
            print(f"{rec['id']:<34} {rec['owned']:<15} {'error: ' + error if error else len(ids)}")

    pkgs = {p: metadata.version(p) for p in ("transformers", "tokenizers", "jinja2", "markupsafe")}
    env = {
        "corpus_format": "starlings.template-corpus",
        "corpus_format_version": 1,
        "generated_by": "model-pipeline/reference/qwen-chat/make_template_corpus.py",
        "renderer": (
            "transformers apply_chat_template (Jinja2 ImmutableSandboxedEnvironment) with the checkpoint template"
        ),
        "profile": args.profile,
        "repository": pin["repository"],
        "revision": args.revision,
        "chat_template_sha256": template_sha,
        "python": platform.python_version(),
        "packages": pkgs,
        "cases": counts,
        "corpus_sha256": sha256_file(corpus),
        "assets": {
            p.name: sha256_file(p)
            for p in sorted(args.model_dir.iterdir())
            if p.is_file() and p.name in {"tokenizer.json", "tokenizer_config.json"}
        },
    }
    (args.out / "environment.json").write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
    print(counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
