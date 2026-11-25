"""Local open-weights models through Ollama's HTTP API.

Same interface as AnthropicLLM: (system, user, schema) -> LLMResult. The JSON
schema of the Pydantic output model is passed as Ollama's `format`, which
constrains generation; the reply is still validated with Pydantic.

Sampling is left at the model's defaults, as it is for Claude (whose
temperature cannot be set), so neither arm gets a hand-tuned temperature.
"""

import json
import urllib.request
from typing import Callable, TypeVar

from pydantic import BaseModel, ValidationError

from echo_sim.llm_policy import LLMResult

T = TypeVar("T", bound=BaseModel)

Post = Callable[[str, dict], dict]

CHARS_PER_TOKEN = 3.5  # a little under what English prompts here use
OUTPUT_RESERVE = 1024  # tokens kept free for the reply


class PromptTooLong(RuntimeError):
    """Raised instead of letting Ollama silently drop the middle of an over-long prompt."""


def _post(url: str, payload: dict, timeout: float = 600.0) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


class OllamaLLM:
    def __init__(
        self,
        model: str = "qwen2.5:7b",
        url: str = "http://localhost:11434",
        num_ctx: int = 8192,  # Ollama silently truncates prompts beyond its default 4096
        post: Post = _post,
    ):
        self.model = model
        self.url = url.rstrip("/")
        self.num_ctx = num_ctx
        self.post = post

    def __call__(self, system: str, user: str, schema: type[T]) -> LLMResult[T]:
        estimate = (len(system) + len(user)) / CHARS_PER_TOKEN
        if estimate > self.num_ctx - OUTPUT_RESERVE:
            raise PromptTooLong(f"~{estimate:.0f} tokens does not fit num_ctx={self.num_ctx} with room to answer")
        data = self.post(
            f"{self.url}/api/chat",
            {
                "model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "format": schema.model_json_schema(),
                "stream": False,
                "options": {"num_ctx": self.num_ctx},  # context size only; sampling stays default
            },
        )
        stop = data.get("done_reason")
        usage = {"input_tokens": data.get("prompt_eval_count", 0), "output_tokens": data.get("eval_count", 0)}
        if stop != "stop":  # e.g. "length": truncated output
            return LLMResult(None, stop, usage, error=f"stopped: {stop}")
        try:
            parsed = schema.model_validate_json(data["message"]["content"])
        except (ValidationError, KeyError) as e:
            return LLMResult(None, stop, usage, error=f"invalid output: {type(e).__name__}")
        return LLMResult(parsed, "end_turn", usage)
