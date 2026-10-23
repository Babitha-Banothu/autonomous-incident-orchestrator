import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from gateway import BudgetExceeded, LLMGateway, tracer

from agents import run_agents
from gateway import BudgetExceeded, LLMGateway
from models import Diagnosis, Finding, ProposedAction
from synthesis import synthesize
from tools import get_recent_deploys

from pathlib import Path
from fastapi.responses import FileResponse

app = FastAPI(title="Incident Investigation Orchestrator")


@app.get("/", include_in_schema=False)
async def dashboard():
    index_file = Path(__file__).resolve().parent / "static" / "index.html"
    return FileResponse(index_file)
    
# In-memory store: incident_id -> record (no database, by design)
INCIDENTS: dict[str, dict] = {}


class InvestigateRequest(BaseModel):
    service: str = "checkout-api"
    alert_time: str = "2026-10-22T14:15:00"


class InvestigateResponse(BaseModel):
    incident_id: str
    findings: list[Finding]
    diagnosis: Diagnosis
    status: str
    cost: dict


def _known_versions(service: str) -> set[str]:
    deploys = get_recent_deploys(service=service, limit=100)
    return {d["version"] for d in deploys} | {d["previous_version"] for d in deploys}


def execute_action(action: ProposedAction) -> str:
    """MOCK executor. Only ever called from /approve."""
    if action.type == "rollback":
        msg = f"[MOCK] Rolled back {action.target_service} to {action.target_version}"
    elif action.type == "restart":
        msg = f"[MOCK] Restarted {action.target_service}"
    else:
        msg = "[MOCK] No action taken"
    print(msg)
    return msg



@app.post("/investigate", response_model=InvestigateResponse)
async def investigate(req: InvestigateRequest = InvestigateRequest()):
    gateway = LLMGateway()

    with tracer.start_as_current_span("investigate") as span:
        span.set_attribute("service", req.service)

        try:
            findings = await run_agents(
                req.service, req.alert_time, gateway
            )
            diagnosis = await synthesize(
                req.service, findings, gateway
            )
        except BudgetExceeded as e:
            raise HTTPException(status_code=429, detail=str(e))

    action = diagnosis.action

    if (
        action.type == "rollback"
        and action.target_version not in _known_versions(action.target_service)
    ):
        raise HTTPException(
            status_code=502,
            detail=f"Unknown rollback target version: {action.target_version}",
        )

    incident_id = uuid.uuid4().hex[:8]

    INCIDENTS[incident_id] = {
        "findings": findings,
        "diagnosis": diagnosis,
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    return InvestigateResponse(
        incident_id=incident_id,
        findings=findings,
        diagnosis=diagnosis,
        status="pending",
        cost=gateway.summary(),
    )

@app.post("/approve/{incident_id}")
async def approve(incident_id: str):
    incident = INCIDENTS.get(incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    if incident["status"] != "pending":
        raise HTTPException(status_code=409, detail=f"Already {incident['status']}")

    result = execute_action(incident["diagnosis"].action)
    incident["status"] = "executed"
    incident["executed_at"] = datetime.now(timezone.utc).isoformat()
    return {"incident_id": incident_id, "status": "executed", "result": result}