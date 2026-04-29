import os
from dataclasses import dataclass
from openai import OpenAI

# ollama:  LLM_BASE_URL=http://localhost:11434/v1  LLM_API_KEY=ollama
# vllm:    LLM_BASE_URL=http://localhost:8000/v1   LLM_API_KEY=token-abc
_client = OpenAI(
    base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"),
    api_key=os.environ.get("LLM_API_KEY", "ollama"),
)


@dataclass
class LLMResponse:
    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    duration_ms: float = 0.0


def chat(model: str, messages: list, temperature: float = 0.0) -> LLMResponse:
    resp = _client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=2048,
    )
    usage = resp.usage
    return LLMResponse(
        content=resp.choices[0].message.content,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
    )
