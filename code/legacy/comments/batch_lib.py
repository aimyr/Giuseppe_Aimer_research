"""Provider-agnostic batch API dispatchers.

Provides:
  submit_batch(judge_alias, requests) -> {provider, job_id, ...metadata}
  check_status(meta) -> "pending" | "completed" | "failed"
  fetch_results(meta) -> dict[custom_id, text]

Each `request` is a dict with keys:
  custom_id: str
  prompt: str
  system: Optional[str]
  temperature: float
  max_tokens: int
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

from judging_lib import (
    JUDGES,
    _anthropic_client,
    _mistral_client,
    _openai_client,
)


HERE = Path(__file__).resolve().parent
JOBS_DIR = HERE / "batch_jobs"


# OpenAI Batch file size limit is ~50k requests / 100MB. We split chunks under that.
OPENAI_MAX_REQUESTS_PER_BATCH = 45_000
# Anthropic Message Batches limit is 100k requests, 256MB.
ANTHROPIC_MAX_REQUESTS_PER_BATCH = 90_000
# Mistral batch is ~10k per file (conservative); split to be safe.
MISTRAL_MAX_REQUESTS_PER_BATCH = 9_000


def _chunked(seq: list, size: int) -> Iterable[list]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

def _openai_request_to_jsonl(req: dict, model_id: str) -> dict:
    """Convert one request to OpenAI Batch JSONL line."""
    if model_id.startswith("gpt-5"):
        # Responses API; omit max_output_tokens so reasoning has full default
        # budget. Capping forces "incomplete" with empty text on longer prompts.
        inp = []
        if req.get("system"):
            inp.append({"role": "system", "content": req["system"]})
        inp.append({"role": "user", "content": req["prompt"]})
        body = {
            "model": model_id,
            "input": inp,
        }
        url = "/v1/responses"
    else:
        # Chat Completions
        messages = []
        if req.get("system"):
            messages.append({"role": "system", "content": req["system"]})
        messages.append({"role": "user", "content": req["prompt"]})
        body = {
            "model": model_id,
            "messages": messages,
            "temperature": float(req.get("temperature", 0.0)),
            "max_tokens": int(req.get("max_tokens", 64)),
        }
        url = "/v1/chat/completions"
    return {
        "custom_id": req["custom_id"],
        "method": "POST",
        "url": url,
        "body": body,
    }


def _submit_openai(model_id: str, requests: list[dict]) -> dict:
    client = _openai_client()
    chunk_jobs: list[dict] = []

    for chunk_idx, chunk in enumerate(_chunked(requests, OPENAI_MAX_REQUESTS_PER_BATCH)):
        # Write JSONL to a temp file and upload.
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False,
                                          encoding="utf-8") as tf:
            for req in chunk:
                tf.write(json.dumps(_openai_request_to_jsonl(req, model_id), ensure_ascii=False))
                tf.write("\n")
            tmp_path = tf.name

        with open(tmp_path, "rb") as fh:
            f = client.files.create(file=fh, purpose="batch")
        Path(tmp_path).unlink(missing_ok=True)

        endpoint = "/v1/responses" if model_id.startswith("gpt-5") else "/v1/chat/completions"
        batch = client.batches.create(
            input_file_id=f.id,
            endpoint=endpoint,
            completion_window="24h",
        )
        chunk_jobs.append({
            "chunk_index": chunk_idx,
            "input_file_id": f.id,
            "batch_id": batch.id,
            "status": getattr(batch, "status", None),
        })
        print(f"  [openai chunk {chunk_idx}] batch_id={batch.id} requests={len(chunk)}")

    return {"provider": "openai", "model_id": model_id, "chunks": chunk_jobs}


def _check_openai(meta: dict) -> str:
    client = _openai_client()
    states = []
    for ck in meta["chunks"]:
        b = client.batches.retrieve(ck["batch_id"])
        states.append(b.status)
        ck["status"] = b.status
    if all(s == "completed" for s in states):
        return "completed"
    if any(s in ("failed", "expired", "cancelled") for s in states):
        return "failed"
    return "pending"


def _fetch_openai(meta: dict) -> dict[str, str]:
    client = _openai_client()
    out: dict[str, str] = {}
    for ck in meta["chunks"]:
        b = client.batches.retrieve(ck["batch_id"])
        if not b.output_file_id:
            continue
        content = client.files.content(b.output_file_id).text
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cid = obj.get("custom_id", "")
            resp = obj.get("response") or {}
            body = resp.get("body") or {}
            text = ""
            if "choices" in body:
                ch = body["choices"][0]
                text = (ch.get("message", {}).get("content") or "").strip()
            elif "output" in body:
                # Responses API output is a list of items
                pieces = []
                for item in body.get("output", []):
                    for part in item.get("content", []):
                        if part.get("type") in ("output_text", "text"):
                            pieces.append(part.get("text", ""))
                text = "".join(pieces).strip()
            out[cid] = text
    return out


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

def _anthropic_request_to_payload(req: dict, model_id: str) -> dict:
    params: dict[str, Any] = {
        "model": model_id,
        "max_tokens": int(req.get("max_tokens", 64)),
        "temperature": float(req.get("temperature", 0.0)),
        "messages": [{"role": "user", "content": req["prompt"]}],
    }
    if req.get("system"):
        params["system"] = req["system"]
    return {
        "custom_id": req["custom_id"],
        "params": params,
    }


def _submit_anthropic(model_id: str, requests: list[dict]) -> dict:
    client = _anthropic_client()
    chunk_jobs: list[dict] = []

    for chunk_idx, chunk in enumerate(_chunked(requests, ANTHROPIC_MAX_REQUESTS_PER_BATCH)):
        payload = [_anthropic_request_to_payload(r, model_id) for r in chunk]
        batch = client.messages.batches.create(requests=payload)
        chunk_jobs.append({
            "chunk_index": chunk_idx,
            "batch_id": batch.id,
            "status": getattr(batch, "processing_status", None),
        })
        print(f"  [anthropic chunk {chunk_idx}] batch_id={batch.id} requests={len(chunk)}")

    return {"provider": "anthropic", "model_id": model_id, "chunks": chunk_jobs}


def _check_anthropic(meta: dict) -> str:
    client = _anthropic_client()
    states = []
    for ck in meta["chunks"]:
        b = client.messages.batches.retrieve(ck["batch_id"])
        states.append(b.processing_status)
        ck["status"] = b.processing_status
    if all(s == "ended" for s in states):
        return "completed"
    return "pending"


def _fetch_anthropic(meta: dict) -> dict[str, str]:
    client = _anthropic_client()
    out: dict[str, str] = {}
    for ck in meta["chunks"]:
        # results() returns an iterator over MessageBatchIndividualResponse
        for item in client.messages.batches.results(ck["batch_id"]):
            cid = item.custom_id
            res = item.result
            if getattr(res, "type", "") == "succeeded" and getattr(res, "message", None):
                pieces = []
                for block in res.message.content:
                    text = getattr(block, "text", None)
                    if text:
                        pieces.append(text)
                out[cid] = "".join(pieces).strip()
            else:
                out[cid] = f"BATCH_ERROR: {getattr(res, 'type', 'unknown')}"
    return out


# ---------------------------------------------------------------------------
# Mistral
# ---------------------------------------------------------------------------

def _mistral_request_to_jsonl(req: dict, model_id: str) -> dict:
    messages = []
    if req.get("system"):
        messages.append({"role": "system", "content": req["system"]})
    messages.append({"role": "user", "content": req["prompt"]})
    return {
        "custom_id": req["custom_id"],
        "body": {
            "model": model_id,
            "messages": messages,
            "temperature": float(req.get("temperature", 0.0)),
            "max_tokens": int(req.get("max_tokens", 64)),
        },
    }


def _submit_mistral(model_id: str, requests: list[dict]) -> dict:
    client = _mistral_client()
    chunk_jobs: list[dict] = []

    for chunk_idx, chunk in enumerate(_chunked(requests, MISTRAL_MAX_REQUESTS_PER_BATCH)):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False,
                                          encoding="utf-8") as tf:
            for req in chunk:
                tf.write(json.dumps(_mistral_request_to_jsonl(req, model_id), ensure_ascii=False))
                tf.write("\n")
            tmp_path = tf.name

        with open(tmp_path, "rb") as fh:
            f = client.files.upload(
                file={"file_name": "batch.jsonl", "content": fh.read()},
                purpose="batch",
            )
        Path(tmp_path).unlink(missing_ok=True)

        job = client.batch.jobs.create(
            input_files=[f.id],
            model=model_id,
            endpoint="/v1/chat/completions",
            metadata={"job_type": "preference"},
        )
        chunk_jobs.append({
            "chunk_index": chunk_idx,
            "input_file_id": f.id,
            "batch_id": job.id,
            "status": getattr(job, "status", None),
        })
        print(f"  [mistral chunk {chunk_idx}] batch_id={job.id} requests={len(chunk)}")

    return {"provider": "mistral", "model_id": model_id, "chunks": chunk_jobs}


def _check_mistral(meta: dict) -> str:
    client = _mistral_client()
    states = []
    for ck in meta["chunks"]:
        j = client.batch.jobs.get(job_id=ck["batch_id"])
        states.append(j.status)
        ck["status"] = j.status
    if all(s == "SUCCESS" for s in states):
        return "completed"
    if any(s in ("FAILED", "CANCELLED", "TIMEOUT_EXCEEDED") for s in states):
        return "failed"
    return "pending"


def _fetch_mistral(meta: dict) -> dict[str, str]:
    client = _mistral_client()
    out: dict[str, str] = {}
    for ck in meta["chunks"]:
        j = client.batch.jobs.get(job_id=ck["batch_id"])
        if not getattr(j, "output_file", None):
            continue
        # client.files.download yields bytes
        raw = client.files.download(file_id=j.output_file).read()
        for line in raw.decode("utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cid = obj.get("custom_id", "")
            body = (obj.get("response") or {}).get("body") or obj.get("response") or {}
            text = ""
            choices = body.get("choices") or []
            if choices:
                text = (choices[0].get("message", {}).get("content") or "").strip()
            out[cid] = text
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

PROVIDER_DISPATCH = {
    "openai":    (_submit_openai,    _check_openai,    _fetch_openai),
    "anthropic": (_submit_anthropic, _check_anthropic, _fetch_anthropic),
    "mistral":   (_submit_mistral,   _check_mistral,   _fetch_mistral),
}


def supports_batch(judge_alias: str) -> bool:
    info = JUDGES.get(judge_alias)
    return info is not None and info["provider"] in PROVIDER_DISPATCH


def submit_batch(judge_alias: str, requests: list[dict]) -> dict:
    info = JUDGES[judge_alias]
    provider = info["provider"]
    if provider not in PROVIDER_DISPATCH:
        raise ValueError(f"No batch support for provider '{provider}' (judge {judge_alias})")
    submit, _, _ = PROVIDER_DISPATCH[provider]
    meta = submit(info["model_id"], requests)
    meta["judge_alias"] = judge_alias
    return meta


def check_status(meta: dict) -> str:
    provider = meta["provider"]
    _, check, _ = PROVIDER_DISPATCH[provider]
    return check(meta)


def fetch_results(meta: dict) -> dict[str, str]:
    provider = meta["provider"]
    _, _, fetch = PROVIDER_DISPATCH[provider]
    return fetch(meta)


def save_meta(meta: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


def load_meta(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
