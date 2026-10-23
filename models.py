from typing import Literal
from pydantic import BaseModel


class Finding(BaseModel):
    agent: str
    summary: str
    severity: Literal["low", "medium", "high"]
    evidence: list[str] = []
    suspected_start: str | None = None


class ProposedAction(BaseModel):
    # Closed set: the LLM cannot invent arbitrary commands
    type: Literal["rollback", "restart", "none"]
    target_service: str
    target_version: str | None = None
    reason: str


class Diagnosis(BaseModel):
    root_cause: str
    confidence: Literal["low", "medium", "high"]
    action: ProposedAction