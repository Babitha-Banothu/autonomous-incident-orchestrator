import os
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv

import re
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

_provider = TracerProvider(resource=Resource.create({"service.name": "incident-orchestrator"}))
_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(_provider)
tracer = trace.get_tracer("orchestrator")

from pathlib import Path

load_dotenv(Path(__file__).resolve().parent / ".env")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Route by task type: cheap model for agents, strong model for synthesis
MODELS = {
    "agent": os.getenv("CHEAP_MODEL", "llama-3.1-8b-instant"),
    "synthesis": os.getenv("STRONG_MODEL", "llama-3.3-70b-versatile"),
}

# USD per 1M tokens (input, output). Notional on the free tier; edit if prices change.
PRICING = {
    "llama-3.1-8b-instant": (0.05, 0.08),
    "llama-3.3-70b-versatile": (0.59, 0.79),
}
DEFAULT_PRICE = (0.59, 0.79)


class BudgetExceeded(RuntimeError):
    """Raised when the total cost cap has been reached."""


@dataclass
class CallRecord:
    task_type: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float



class LLMGateway:
    """One instance per investigation, so the cap applies per investigation."""

    def __init__(self, max_cost_usd: float | None = None):
        self.max_cost_usd = (
            max_cost_usd
            if max_cost_usd is not None
            else float(os.getenv("MAX_COST_USD", "0.05"))
        )
        self.calls: list[CallRecord] = []

    @property
    def total_cost(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    async def complete(
        self,
        task_type: str,
        system: str,
        user: str,
        json_mode: bool = False,
    ) -> str:
        with tracer.start_as_current_span("llm.call") as span:
            span.set_attribute("llm.task_type", task_type)

            if task_type not in MODELS:
                raise ValueError(f"Unknown task_type: {task_type}")

            if self.total_cost >= self.max_cost_usd:
                raise BudgetExceeded(
                    f"Cost cap reached: ${self.total_cost:.6f} >= "
                    f"${self.max_cost_usd:.6f}"
                )

            key = os.getenv("GROQ_API_KEY")
            if not key:
                raise RuntimeError("GROQ_API_KEY missing. Add it to .env")

            model = MODELS[task_type]
            span.set_attribute("llm.model", model)

            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.2,
            }

            if json_mode:
                payload["response_format"] = {"type": "json_object"}

            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    GROQ_URL,
                    headers={"Authorization": f"Bearer {key}"},
                    json=payload,
                )

            resp.raise_for_status()
            data = resp.json()

            usage = data["usage"]
            p_in, p_out = PRICING.get(model, DEFAULT_PRICE)
            cost = (
                usage["prompt_tokens"] * p_in
                + usage["completion_tokens"] * p_out
            ) / 1_000_000

            self.calls.append(
                CallRecord(
                    task_type,
                    model,
                    usage["prompt_tokens"],
                    usage["completion_tokens"],
                    cost,
                )
            )

            span.set_attribute("llm.input_tokens", usage["prompt_tokens"])
            span.set_attribute("llm.output_tokens", usage["completion_tokens"])
            span.set_attribute("llm.cost_usd", cost)
            span.set_attribute("llm.total_cost_usd", self.total_cost)

            return data["choices"][0]["message"]["content"]

    def summary(self) -> dict:
        return {
            "total_cost_usd": round(self.total_cost, 6),
            "cap_usd": self.max_cost_usd,
            "calls": len(self.calls),
        }


    # ---- Prompt-injection guard ----
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"disregard\s+(all\s+)?(previous|prior|above)",
    r"(reveal|show|print)\s+(your\s+)?(system\s+)?prompt",
    r"\bsystem\s*prompt\b",
    r"you\s+are\s+now\b",
    r"\bnew\s+instructions?\b",
    r"(delete|drop|wipe)\s+(the\s+)?(production\s+)?(database|db|tables?)",
]
_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]
REDACTED = "[REDACTED: suspected prompt injection]"


def sanitize(text: str) -> tuple[str, bool]:
    """Returns (safe_text, was_flagged). A flagged message is replaced entirely."""
    if any(rx.search(text) for rx in _INJECTION_RE):
        return REDACTED, True
    return text, False