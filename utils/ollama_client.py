import requests
from dataclasses import dataclass

OLLAMA_URL = "http://localhost:11434/api/chat"


@dataclass
class OllamaResponse:
    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    duration_ms: float


def chat(model: str, messages: list, temperature: float = 0.0) -> OllamaResponse:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 2048},
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=600)
    resp.raise_for_status()
    data = resp.json()

    prompt_tokens = data.get("prompt_eval_count", 0)
    completion_tokens = data.get("eval_count", 0)
    duration_ms = data.get("eval_duration", 0) / 1_000_000

    return OllamaResponse(
        content=data["message"]["content"],
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        duration_ms=duration_ms,
    )
