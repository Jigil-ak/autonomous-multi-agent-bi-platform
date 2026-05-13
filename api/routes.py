"""
FastAPI Routes — Multi-Agent BI Platform
=========================================
Exposes four REST endpoints for the Streamlit dashboard and external clients.

Endpoints:
    POST /analyze           → launch workflow, return task_id immediately
    GET  /status/{task_id}  → poll workflow progress
    GET  /logs/{task_id}    → incremental execution logs
    GET  /report/{task_id}  → final consolidated report

Design:
- BackgroundTasks keeps POST /analyze non-blocking (202 Accepted)
- TASK_REGISTRY is a thread-safe in-process store (no Redis/Celery)
- WorkflowResult is serialised to JSON and persisted in ./reports/
- Fallback degradation produces COMPLETED_WITH_WARNINGS, not FAILED
"""

from __future__ import annotations

import json
import os
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from agents.orchestrator import run_workflow
from agents.schemas import WorkflowInput, WorkflowStage
from utils.logger import get_logger

logger = get_logger("api.routes")

router = APIRouter()

# ---------------------------------------------------------------------------
# Persistence directory
# ---------------------------------------------------------------------------
REPORTS_DIR = Path("./reports")
REPORTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Task Registry — thread-safe in-process store
# ---------------------------------------------------------------------------
_registry_lock = threading.Lock()

TASK_REGISTRY: dict[str, dict[str, Any]] = {}


def _new_task_record(task_id: str, workflow_input: WorkflowInput) -> dict[str, Any]:
    """Create a blank task record for a new workflow."""
    return {
        "task_id": task_id,
        "status": "PENDING",
        "stage": "initializing",
        "created_at": time.time(),
        "updated_at": time.time(),
        "completed_at": None,
        "progress_percent": 0,
        "active_agent": None,
        "completed_agents": [],
        "failed_agents": [],
        "fallback_events": [],
        "logs": [],
        "warnings": [],
        "token_estimate": 0,
        "report": None,
        "workflow_input": workflow_input.model_dump(),
        "error": None,
    }


def _update_registry(task_id: str, updates: dict[str, Any]) -> None:
    with _registry_lock:
        if task_id in TASK_REGISTRY:
            TASK_REGISTRY[task_id].update(updates)
            TASK_REGISTRY[task_id]["updated_at"] = time.time()


def _append_log(task_id: str, message: str) -> None:
    with _registry_lock:
        if task_id in TASK_REGISTRY:
            TASK_REGISTRY[task_id]["logs"].append({
                "ts": time.strftime("%H:%M:%S"),
                "msg": message,
            })
            TASK_REGISTRY[task_id]["updated_at"] = time.time()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class AnalyzeRequest(BaseModel):
    company_description: str
    product_details: str
    target_audience: str
    goals: str
    constraints: str = "None specified"
    enable_human_approval: bool = False


class AnalyzeResponse(BaseModel):
    task_id: str
    status: str
    message: str


class StatusResponse(BaseModel):
    task_id: str
    status: str
    stage: str
    progress_percent: int
    active_agent: Optional[str]
    completed_agents: list[str]
    failed_agents: list[str]
    fallback_events: list[str]
    warnings: list[str]
    token_estimate: int
    created_at: float
    updated_at: float
    error: Optional[str]


class LogsResponse(BaseModel):
    task_id: str
    logs: list[dict[str, str]]
    total_entries: int


# ---------------------------------------------------------------------------
# Background workflow runner
# ---------------------------------------------------------------------------
def _run_workflow_background(
    task_id: str,
    workflow_input: WorkflowInput,
    enable_human_approval: bool,
) -> None:
    """
    Execute the multi-agent workflow in a background thread.
    Updates TASK_REGISTRY at each stage so the frontend can poll progress.
    """
    _update_registry(task_id, {"status": "RUNNING", "stage": "initializing"})
    _append_log(task_id, "Workflow started. Initializing agents...")

    try:
        # ----------------------------------------------------------------
        # Progress callback to relay real-time state to TASK_REGISTRY
        # ----------------------------------------------------------------
        _seen_agents: set[str] = set()
        
        def _on_progress(state: WorkflowState):
            _stage_to_agent_id = {
                WorkflowStage.RESEARCHING:  "research_agent",
                WorkflowStage.STRATEGIZING: "strategy_agent",
                WorkflowStage.PLANNING:     "planner_agent",
                WorkflowStage.CRITIQUING:   "critic_agent",
                WorkflowStage.QA_CHECK:     "qa_agent",
                WorkflowStage.STORING:      "memory_agent",
            }
            stage_str = state.stage.value if state.stage else "initializing"
            agent_id = _stage_to_agent_id.get(state.stage)
            
            # Log agent start dynamically
            if agent_id and agent_id not in _seen_agents:
                _seen_agents.add(agent_id)
                agent_name = agent_id.replace("_", " ").title()
                _append_log(task_id, f"▶ {agent_name} started...")
                
            # Update registry synchronously
            _update_registry(task_id, {
                "stage": stage_str,
                "active_agent": agent_id,
                "completed_agents": state.completed_agents.copy() if hasattr(state, 'completed_agents') else [],
                "failed_agents": state.failed_agents.copy() if hasattr(state, 'failed_agents') else [],
                "token_estimate": getattr(state, 'total_tokens', 0)
            })

        # ----------------------------------------------------------------
        # Execute workflow
        # ----------------------------------------------------------------
        result = run_workflow(
            workflow_input, 
            enable_human_approval=enable_human_approval,
            progress_callback=_on_progress
        )

        # ----------------------------------------------------------------
        # Derive final status
        # ----------------------------------------------------------------
        has_failures = bool(result.workflow_state and result.workflow_state.failed_agents)
        has_mock = any([
            result.research and result.research.is_mock,
            result.strategy and result.strategy.is_mock,
            result.execution_plan and result.execution_plan.is_mock,
        ])
        has_fallback = result.workflow_state and result.workflow_state.total_tokens > 0

        # Detect fallback events from logs
        fallback_events: list[str] = []
        if result.research and not result.research.is_mock:
            # Check for fallback in model router logs (read from file)
            try:
                log_path = Path("logs/platform.log")
                if log_path.exists():
                    recent = log_path.read_text(encoding="utf-8").splitlines()[-200:]
                    for line in recent:
                        if "fallback" in line.lower() and task_id in line:
                            model_hint = "Flash (fallback from Pro)"
                            if model_hint not in fallback_events:
                                fallback_events.append(model_hint)
                        elif "status=fallback" in line:
                            fb_msg = "Gemini Flash used (Pro quota fallback)"
                            if fb_msg not in fallback_events:
                                fallback_events.append(fb_msg)
            except Exception:
                pass

        if has_failures or not result.success:
            final_status = "FAILED"
        elif has_mock or fallback_events:
            final_status = "COMPLETED_WITH_WARNINGS"
        else:
            final_status = "COMPLETED"

        # ----------------------------------------------------------------
        # Serialise report
        # ----------------------------------------------------------------
        report_data = _serialise_result(result)

        # Persist to disk
        report_path = REPORTS_DIR / f"{task_id}.json"
        report_path.write_text(
            json.dumps(report_data, indent=2, default=str),
            encoding="utf-8",
        )
        logger.info(f"Report persisted: {report_path}")

        # ----------------------------------------------------------------
        # Build warnings list
        # ----------------------------------------------------------------
        warnings: list[str] = []
        if has_mock:
            warnings.append("Some agent outputs used mock/fallback data.")
        if fallback_events:
            warnings.append("Gemini 2.5 Pro quota exhausted — Flash fallback activated.")
        if has_failures:
            warnings.append(f"Agents failed: {result.workflow_state.failed_agents}")

        ws = result.workflow_state
        _update_registry(task_id, {
            "status": final_status,
            "stage": result.workflow_state.stage.value if result.workflow_state else "completed",
            "progress_percent": 100,
            "completed_at": time.time(),
            "completed_agents": ws.completed_agents if ws else [],
            "failed_agents": ws.failed_agents if ws else [],
            "fallback_events": fallback_events,
            "warnings": warnings,
            "token_estimate": ws.total_tokens if ws else 0,
            "report": report_data,
            "active_agent": None,
        })
        _append_log(task_id, f"✅ Workflow complete. Status: {final_status}")
        if warnings:
            for w in warnings:
                _append_log(task_id, f"⚠ {w}")

    except Exception as exc:
        _running["active"] = False
        error_msg = f"Workflow exception: {exc}"
        tb = traceback.format_exc()
        logger.error(f"[routes] {error_msg}\n{tb}")
        _update_registry(task_id, {
            "status": "FAILED",
            "stage": "failed",
            "error": error_msg,
            "completed_at": time.time(),
        })
        _append_log(task_id, f"❌ Workflow failed: {exc}")


def _serialise_result(result: Any) -> dict[str, Any]:
    """Convert WorkflowResult to a JSON-serialisable dict."""
    try:
        return result.model_dump(mode="json")
    except Exception:
        return {"error": "Could not serialise result", "task_id": str(result.task_id)}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.post("/analyze", response_model=AnalyzeResponse, status_code=202)
async def analyze(
    request: AnalyzeRequest,
    background_tasks: BackgroundTasks,
) -> AnalyzeResponse:
    """
    Start a new multi-agent workflow.

    Returns 202 Accepted immediately with a task_id.
    The workflow runs asynchronously in a background thread.
    """
    task_id = str(uuid.uuid4())[:8]

    workflow_input = WorkflowInput(
        company_description=request.company_description,
        product_details=request.product_details,
        target_audience=request.target_audience,
        goals=request.goals,
        constraints=request.constraints,
        task_id=task_id,
    )

    # Register task before launching background job
    with _registry_lock:
        TASK_REGISTRY[task_id] = _new_task_record(task_id, workflow_input)

    logger.info(f"[routes] New workflow task_id={task_id} registered.")

    background_tasks.add_task(
        _run_workflow_background,
        task_id,
        workflow_input,
        request.enable_human_approval,
    )

    return AnalyzeResponse(
        task_id=task_id,
        status="PENDING",
        message=f"Workflow started. Poll GET /status/{task_id} for updates.",
    )


@router.get("/status/{task_id}", response_model=StatusResponse)
async def get_status(task_id: str) -> StatusResponse:
    """Return current workflow execution status."""
    # Try in-memory first
    with _registry_lock:
        record = TASK_REGISTRY.get(task_id)

    # Try disk if not in memory (replay mode)
    if not record:
        report_path = REPORTS_DIR / f"{task_id}.json"
        if report_path.exists():
            try:
                data = json.loads(report_path.read_text(encoding="utf-8"))
                # Build a minimal status from saved report
                return StatusResponse(
                    task_id=task_id,
                    status="COMPLETED",
                    stage="completed",
                    progress_percent=100,
                    active_agent=None,
                    completed_agents=[],
                    failed_agents=[],
                    fallback_events=[],
                    warnings=["Loaded from saved report (replay mode)."],
                    token_estimate=0,
                    created_at=0,
                    updated_at=time.time(),
                    error=None,
                )
            except Exception:
                pass
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found.")

    # Derive live progress from workflow stage
    stage_progress = {
        "initializing": 5, "researching": 20, "strategizing": 40,
        "planning": 60, "critiquing": 75, "qa_check": 88,
        "storing": 95, "completed": 100, "failed": 100,
    }
    stage = record.get("stage", "initializing")
    progress = record.get("progress_percent") or stage_progress.get(stage, 0)

    # Derive active agent from stage
    stage_agent = {
        "researching": "research_agent", "strategizing": "strategy_agent",
        "planning": "planner_agent", "critiquing": "critic_agent",
        "qa_check": "qa_agent", "storing": "memory_agent",
    }
    active = record.get("active_agent") or stage_agent.get(stage)

    return StatusResponse(
        task_id=task_id,
        status=record.get("status", "PENDING"),
        stage=stage,
        progress_percent=progress,
        active_agent=active,
        completed_agents=record.get("completed_agents", []),
        failed_agents=record.get("failed_agents", []),
        fallback_events=record.get("fallback_events", []),
        warnings=record.get("warnings", []),
        token_estimate=record.get("token_estimate", 0),
        created_at=record.get("created_at", 0),
        updated_at=record.get("updated_at", 0),
        error=record.get("error"),
    )


@router.get("/logs/{task_id}", response_model=LogsResponse)
async def get_logs(task_id: str) -> LogsResponse:
    """Return execution logs for a task."""
    with _registry_lock:
        record = TASK_REGISTRY.get(task_id)

    if not record:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found.")

    logs = record.get("logs", [])
    return LogsResponse(
        task_id=task_id,
        logs=logs,
        total_entries=len(logs),
    )


@router.get("/report/{task_id}")
async def get_report(task_id: str) -> dict[str, Any]:
    """Return the final consolidated report for a completed task."""
    # Check in-memory first
    with _registry_lock:
        record = TASK_REGISTRY.get(task_id)

    if record and record.get("report"):
        return {"task_id": task_id, "report": record["report"]}

    # Try disk (replay)
    report_path = REPORTS_DIR / f"{task_id}.json"
    if report_path.exists():
        try:
            data = json.loads(report_path.read_text(encoding="utf-8"))
            return {"task_id": task_id, "report": data}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to load report: {exc}")

    if record and record.get("status") in ("PENDING", "RUNNING"):
        raise HTTPException(status_code=202, detail="Workflow still running. Try again later.")

    raise HTTPException(status_code=404, detail=f"Report for task {task_id} not found.")


@router.get("/tasks")
async def list_tasks() -> dict[str, Any]:
    """List all registered tasks (recent runs)."""
    with _registry_lock:
        tasks = [
            {
                "task_id": tid,
                "status": r.get("status"),
                "stage": r.get("stage"),
                "created_at": r.get("created_at"),
                "completed_at": r.get("completed_at"),
            }
            for tid, r in TASK_REGISTRY.items()
        ]

    # Also check saved reports on disk
    saved = []
    for path in REPORTS_DIR.glob("*.json"):
        tid = path.stem
        if tid not in [t["task_id"] for t in tasks]:
            saved.append({
                "task_id": tid,
                "status": "COMPLETED",
                "stage": "completed",
                "source": "disk",
            })

    return {
        "active_tasks": tasks,
        "saved_reports": saved,
        "total": len(tasks) + len(saved),
    }
