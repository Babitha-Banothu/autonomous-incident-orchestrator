
import json
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent / "data"


class ToolAccessError(PermissionError):
    """Raised when an agent calls a tool outside its allowlist."""


def _load(name: str) -> Any:
    path = DATA_DIR / f"{name}.json"

    if not path.is_file():
        raise FileNotFoundError(f"Mock data file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_metrics(service: str, metric: str) -> list[dict]:
    data = _load("metrics")

    if data.get("service") != service:
        return []

    return data.get("metrics", {}).get(metric, [])


def search_logs(
    query: str = "",
    level: str | None = None,
    limit: int = 50,
) -> list[dict]:
    if not 1 <= limit <= 500:
        raise ValueError("limit must be between 1 and 500")

    logs = _load("logs")

    if level:
        logs = [log for log in logs if log.get("level") == level.upper()]

    if query:
        logs = [
            log for log in logs
            if query.lower() in log.get("message", "").lower()
        ]

    return logs[:limit]


def get_recent_deploys(
    service: str | None = None,
    limit: int = 5,
) -> list[dict]:
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")

    deploys = _load("deploys")

    if service:
        deploys = [
            deploy for deploy in deploys
            if deploy.get("service") == service
        ]

    return sorted(
        deploys,
        key=lambda deploy: deploy["ts"],
        reverse=True,
    )[:limit]


TOOLS = {
    "get_metrics": get_metrics,
    "search_logs": search_logs,
    "get_recent_deploys": get_recent_deploys,
}

ALLOWLIST = {
    "metrics_agent": {"get_metrics"},
    "logs_agent": {"search_logs"},
    "deploy_agent": {"get_recent_deploys"},
}


def call_tool(agent: str, tool: str, **kwargs):
    allowed_tools = ALLOWLIST.get(agent)

    if allowed_tools is None:
        raise ToolAccessError(f"Unknown agent: {agent}")

    if tool not in allowed_tools:
        raise ToolAccessError(f"{agent} is not allowed to call {tool}")

    function = TOOLS.get(tool)

    if function is None:
        raise ToolAccessError(f"Unknown tool: {tool}")

    return function(**kwargs)
