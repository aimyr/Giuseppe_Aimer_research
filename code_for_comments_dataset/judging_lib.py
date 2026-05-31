"""Shared utilities for multi-model judging.

Provider routing: OpenAI (gpt5nano, gpt4omini), Anthropic (sonnet45),
Together (llama, qwen, mixtral), Mistral (mistral).

All judges expose a single function: query_judge(model_alias, prompt, *,
temperature, system, max_tokens) -> str.
"""

from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv  # type: ignore
    # Look for .env in clean_3rd_dataset/ (parent of judging/).
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass


JUDGES = {
    "gpt5nano":  {"provider": "openai",    "model_id": "gpt-5-nano"},
    "gpt4omini": {"provider": "openai",    "model_id": "gpt-4o-mini"},
    "sonnet45":  {"provider": "anthropic", "model_id": "claude-sonnet-4-5"},
    "llama":     {"provider": "together",  "model_id": "aymerkoshmambe_747d/meta-llama/Meta-Llama-3-8B-Instruct-1885aa25"},
    "qwen":      {"provider": "together",  "model_id": "aymerkoshmambe_747d/Qwen/Qwen2.5-7B-Instruct-Turbo-b83348ca"},
    "mistral":   {"provider": "together",  "model_id": "aymerkoshmambe_747d/mistralai/Mixtral-8x7B-Instruct-v0.1-75df7cc3"},
}


def list_judges() -> list[str]:
    return list(JUDGES.keys())


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        print(f"ERROR: missing env var {name} (looked in shell + clean_3rd_dataset/.env)",
              file=sys.stderr)
        sys.exit(2)
    return val


# Lazy-init clients so importing judging_lib doesn't require every provider's key.
_clients: dict[str, object] = {}


def _openai_client():
    if "openai" not in _clients:
        from openai import OpenAI
        _clients["openai"] = OpenAI(api_key=_require_env("OPENAI_API_KEY"))
    return _clients["openai"]


def _anthropic_client():
    if "anthropic" not in _clients:
        from anthropic import Anthropic
        _clients["anthropic"] = Anthropic(api_key=_require_env("ANTHROPIC_API_KEY"))
    return _clients["anthropic"]


def _together_client():
    if "together" not in _clients:
        from together import Together
        _clients["together"] = Together(api_key=_require_env("TOGETHER_API_KEY"))
    return _clients["together"]


def _mistral_client():
    if "mistral" not in _clients:
        # mistralai 2.x exposes Mistral at mistralai.client.sdk; 1.x at top-level.
        try:
            from mistralai import Mistral  # 1.x
        except ImportError:
            from mistralai.client.sdk import Mistral  # 2.x
        _clients["mistral"] = Mistral(api_key=_require_env("MISTRAL_API_KEY"))
    return _clients["mistral"]


def _call_openai(model_id: str, prompt: str, *, temperature: float,
                 system: Optional[str], max_tokens: int) -> str:
    client = _openai_client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    # gpt-5-nano uses Responses API only; other gpt-* still work via chat.completions.
    if model_id.startswith("gpt-5"):
        # No max_output_tokens — reasoning models need headroom that caller
        # can't size correctly; defaults work better than a capped budget.
        resp = client.responses.create(
            model=model_id,
            input=([{"role": "system", "content": system}] if system else []) +
                  [{"role": "user", "content": prompt}],
        )
        return (resp.output_text or "").strip()

    kwargs = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = client.chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip()


def _call_anthropic(model_id: str, prompt: str, *, temperature: float,
                    system: Optional[str], max_tokens: int) -> str:
    client = _anthropic_client()
    kwargs = {
        "model": model_id,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system
    resp = client.messages.create(**kwargs)
    # Anthropic returns a list of content blocks.
    parts = []
    for block in resp.content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "".join(parts).strip()


def _call_together(model_id: str, prompt: str, *, temperature: float,
                   system: Optional[str], max_tokens: int) -> str:
    client = _together_client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def _call_mistral(model_id: str, prompt: str, *, temperature: float,
                  system: Optional[str], max_tokens: int) -> str:
    client = _mistral_client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.complete(
        model=model_id,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def query_judge(
    model_alias: str,
    prompt: str,
    *,
    temperature: float = 0.0,
    system: Optional[str] = None,
    max_tokens: int = 64,
    max_retries: int = 4,
) -> str:
    if model_alias not in JUDGES:
        raise ValueError(f"Unknown judge alias: {model_alias}. Known: {list_judges()}")
    info = JUDGES[model_alias]
    provider = info["provider"]
    model_id = info["model_id"]
    caller = {
        "openai": _call_openai,
        "anthropic": _call_anthropic,
        "together": _call_together,
        "mistral": _call_mistral,
    }[provider]

    delay = 1.5
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            return caller(model_id, prompt,
                          temperature=temperature, system=system, max_tokens=max_tokens)
        except Exception as exc:
            last_exc = exc
            if attempt == max_retries:
                break
            sleep_for = min(30.0, delay) + random.uniform(0, 0.5)
            print(f"  [{model_alias}] retry {attempt}/{max_retries} after {sleep_for:.1f}s ({exc})",
                  file=sys.stderr)
            time.sleep(sleep_for)
            delay *= 2
    return f"ERROR: {type(last_exc).__name__}: {last_exc}"
