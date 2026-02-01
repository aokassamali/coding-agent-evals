# What this module does:
# Provide a tiny OpenAI-compatible chat client for llama.cpp's llama-server.
# We call /v1/chat/completions and extract the assistant message content.

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import os
import requests


@dataclass
class LLMConfig:
    base_url: str = "http://127.0.0.1:8080/v1"
    model: str = "local-model"  # llama-server ignores this if only one model is loaded
    temperature: float = 0.0
    max_tokens: int = 512
    timeout_s: int = 120

def fetch_server_models(cfg: "LLMConfig") -> Dict[str, Any]:
    """
    Returns raw JSON from GET {base_url}/models (OpenAI-style).
    If unsupported, returns {"error": "..."}.
    """
    url = cfg.base_url.rstrip("/") + "/models"
    try:
        r = requests.get(url, timeout=min(10, cfg.timeout_s))
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "url": url}

def chat(messages: List[Dict[str, str]], cfg: Optional[LLMConfig] = None) -> str:
    cfg = cfg or LLMConfig(base_url=os.getenv("LLAMA_BASE_URL", "http://127.0.0.1:8080/v1"))

    url = cfg.base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": cfg.model,
        "messages": messages,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
    }

    r = requests.post(url, json=payload, timeout=cfg.timeout_s)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]
