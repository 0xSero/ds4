#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request


def make_long_prompt(tokens_hint: int) -> tuple[str, str]:
    marker = f"SPARK-CTX-{tokens_hint}-OMEGA"
    filler = (
        "calibration prefill routing cache tensor benchmark local inference "
        "context retention sparse expert validation sentence "
    )
    # The filler is intentionally wordy; DS4 tokenizes it denser than the rough
    # word count. Keep the generated prompt below the requested frontier so a
    # 200K check leaves room for the answer.
    reps = max(1, tokens_hint // 23)
    lines = []
    insert_at = max(1, reps * 3 // 4)
    for i in range(reps):
        if i == insert_at:
            lines.append(f"Important hidden marker: {marker}. Remember this exact marker.")
        lines.append(f"Section {i:06d}: {filler}")
    lines.append(f"What is the exact hidden marker? Reply with only the marker string.")
    return "\n".join(lines), marker


def count_rough_tokens(text: str) -> int:
    return max(1, len(re.findall(r"\S+", text)))


def merge_tool_call(tool_calls: dict[int, dict], incoming: dict) -> None:
    index = incoming.get("index", 0)
    current = tool_calls.setdefault(
        index,
        {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
    )
    if incoming.get("id"):
        current["id"] += incoming["id"]
    if incoming.get("type"):
        current["type"] = incoming["type"]
    fn = incoming.get("function") or {}
    if fn.get("name"):
        current["function"]["name"] += fn["name"]
    if fn.get("arguments"):
        current["function"]["arguments"] += fn["arguments"]


def stream_chat(base_url: str, model: str, messages: list[dict], max_tokens: int, tools: list[dict] | None) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "think": False,
        "thinking": {"type": "disabled"},
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    start = time.perf_counter()
    first_token_at = None
    chunks: list[str] = []
    reasoning_chunks: list[str] = []
    tool_calls: dict[int, dict] = {}
    usage = None
    url = base_url.rstrip("/") + "/v1/chat/completions"

    for attempt in range(2):
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            resp_ctx = urllib.request.urlopen(req, timeout=1800)
            break
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if attempt == 0 and exc.code in {400, 422} and "stream_options" in body:
                payload.pop("stream_options", None)
                continue
            print(body)
            raise

    with resp_ctx as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            event = json.loads(data)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("delta") or {}
                for tool_call in delta.get("tool_calls") or []:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    merge_tool_call(tool_calls, tool_call)
                reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
                if reasoning:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    reasoning_chunks.append(reasoning)
                text = delta.get("content") or ""
                if text:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    chunks.append(text)

    end = time.perf_counter()
    text = "".join(chunks)
    reasoning_text = "".join(reasoning_chunks)
    generation_s = (end - first_token_at) if first_token_at else None
    prompt_tokens = (usage or {}).get("prompt_tokens")
    completion_tokens = (usage or {}).get("completion_tokens")
    rough_completion_tokens = count_rough_tokens(text + " " + reasoning_text)
    return {
        "ttft_s": (first_token_at - start) if first_token_at else None,
        "wall_s": end - start,
        "generation_s": generation_s,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": (usage or {}).get("total_tokens"),
        "prefill_tokens_per_s": (prompt_tokens / (first_token_at - start)) if prompt_tokens and first_token_at and first_token_at > start else None,
        "decode_tokens_per_s": (completion_tokens / generation_s) if completion_tokens and generation_s and generation_s > 0 else None,
        "rough_completion_tokens": rough_completion_tokens,
        "rough_decode_tokens_per_s": (rough_completion_tokens / generation_s) if generation_s and generation_s > 0 else None,
        "usage": usage,
        "tool_calls": [tool_calls[i] for i in sorted(tool_calls)],
        "reasoning_chars": len(reasoning_text),
        "reasoning_text": reasoning_text,
        "text": text,
    }


def task_payload(task: str, context_tokens: int) -> tuple[list[dict], list[dict] | None, str | None]:
    if task == "long":
        prompt, marker = make_long_prompt(context_tokens)
        return [{"role": "user", "content": prompt}], None, marker
    if task == "tool":
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]
        return [{"role": "user", "content": "Call the weather tool for Tokyo. Do not answer in prose."}], tools, None
    prompts = {
        "smoke": "Say exactly: REAP online.",
        "code": "Explain this Python code and identify edge cases: def f(xs): return xs[0] if xs else None",
        "mermaid": "Create only a Mermaid flowchart explaining TTFT, prefill, decode, and KV cache flow.",
        "unicode": "Create only a Unicode box-drawing diagram explaining sparse MoE routing.",
        "religion": "Compare Christianity, Islam, and Buddhism on ultimate reality in a neutral, respectful way.",
        "philosophy": "Compare apophatic theology and Buddhist emptiness in four precise bullets.",
    }
    return [{"role": "user", "content": prompts.get(task, task)}], None, None


def evaluate(task: str, result: dict, marker: str | None) -> dict:
    text = result["text"]
    reasoning_text = result.get("reasoning_text") or ""
    combined = text + "\n" + reasoning_text
    lowered = text.lower()
    tool_calls = result["tool_calls"]
    checks: dict[str, bool | int | str] = {
        "text_chars": len(text),
        "reasoning_chars": len(reasoning_text),
        "tool_call_count": len(tool_calls),
    }
    if task == "long":
        checks["marker"] = marker or ""
        checks["needle_retained_visible"] = bool(marker and marker in text)
        checks["needle_retained_reasoning"] = bool(marker and marker in reasoning_text)
        checks["needle_retained"] = bool(marker and marker in combined)
        checks["passed_core"] = bool(checks["needle_retained_visible"])
    elif task == "tool":
        names = [((tc.get("function") or {}).get("name") or "") for tc in tool_calls]
        args = [((tc.get("function") or {}).get("arguments") or "") for tc in tool_calls]
        checks["called_get_weather"] = "get_weather" in names
        checks["arguments_mention_tokyo"] = any("tokyo" in arg.lower() for arg in args)
        checks["passed_core"] = bool(checks["called_get_weather"] and checks["arguments_mention_tokyo"])
    elif task == "smoke":
        checks["passed_core"] = "reap online" in lowered
    elif task == "code":
        checks["passed_core"] = "none" in lowered and ("empty" in lowered or "[]" in text)
    elif task == "mermaid":
        checks["passed_core"] = "flowchart" in lowered or "graph " in lowered
    elif task == "unicode":
        checks["passed_core"] = any(ch in text for ch in "┌┐└┘│─")
    elif task == "religion":
        checks["passed_core"] = all(term in lowered for term in ["christ", "islam", "buddh"])
    elif task == "philosophy":
        checks["passed_core"] = "apophatic" in lowered and ("emptiness" in lowered or "sunyata" in lowered or "śūnyatā" in lowered)
    else:
        checks["passed_core"] = len(text) > 0
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", default="smoke")
    parser.add_argument("--context-tokens", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    messages, tools, marker = task_payload(args.task, args.context_tokens)
    result = stream_chat(args.base_url, args.model, messages, args.max_tokens, tools)
    result.update(
        {
            "base_url": args.base_url,
            "model": args.model,
            "task": args.task,
            "context_tokens_hint": args.context_tokens,
            "max_tokens": args.max_tokens,
            "prompt_chars": sum(len(str(m.get("content", ""))) for m in messages),
            "created_unix": time.time(),
        }
    )
    result["eval"] = evaluate(args.task, result, marker)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")
    summary = {k: v for k, v in result.items() if k not in {"text"}}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if result["eval"].get("passed_core") else 4


if __name__ == "__main__":
    raise SystemExit(main())
