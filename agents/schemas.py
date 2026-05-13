"""
Pydantic V2 Schemas for Inter-Agent Communication
===================================================
Centralized schema definitions for all agent inputs and outputs.

All schemas are:
- Pydantic V2 compatible
- JSON-serializable
- Reusable across agents and API responses
- Validated before passing data downstream

Schema hierarchy:
  WorkflowInput          → provided by user / API
  ResearchFindings       → output of Research Agent
  StrategyOutput         → output of Strategy Agent
  ExecutionPlan          → output of Planner Agent
  CritiqueOutput         → output of Critic Agent
  QAReport               → output of QA Agent
  MemoryEntry            → stored in ChromaDB
  ExecutionTrace         → per-agent observability record
  WorkflowState          → full workflow tracking object
  WorkflowResult         → final consolidated report
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class AgentStatus(str, Enum):
    """Possible states for an agent during workflow execution."""
    PENDING   = "pending"
    RUNNING   = "running"
    SUCCESS   = "success"
    FAILED    = "failed"
    SKIPPED   = "skipped"
    FALLBACK  = "fallback"


class WorkflowStage(str, Enum):
    """High-level stages of the platform workflow."""
    INITIALIZING = "initializing"
    RESEARCHING  = "researching"
    STRATEGIZING = "strategizing"
    PLANNING     = "planning"
    CRITIQUING   = "critiquing"
    QA_CHECK     = "qa_check"
    STORING      = "storing"
    COMPLETED    = "completed"
    FAILED       = "failed"


# ---------------------------------------------------------------------------
# Input Schema
# ---------------------------------------------------------------------------

class WorkflowInput(BaseModel):
    """User-provided business context that seeds the entire workflow."""

    company_description: str = Field(
        ...,
        min_length=10,
        description="Brief description of the company or startup.",
    )
    product_details: str = Field(
        ...,
        min_length=10,
        description="Description of the product or service being analyzed.",
    )
    target_audience: str = Field(
        ...,
        min_length=5,
        description="Target customer segment or market.",
    )
    goals: str = Field(
        ...,
        min_length=5,
        description="Primary business goals or objectives.",
    )
    constraints: str = Field(
        default="None specified",
        description="Known constraints — budget, timeline, team size, etc.",
    )
    task_id: Optional[str] = Field(
        default=None,
        description="Unique task identifier assigned by the API.",
    )

    model_config = {"str_strip_whitespace": True}


# ---------------------------------------------------------------------------
# Research Agent Output
# ---------------------------------------------------------------------------

class Competitor(BaseModel):
    """A single competitor entry."""
    name: str
    description: str = ""
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    pricing: str = ""


class ResearchFindings(BaseModel):
    """Structured output from the Research Agent."""

    competitors: list[Competitor] = Field(
        default_factory=list,
        description="List of identified competitors.",
    )
    market_signals: list[str] = Field(
        default_factory=list,
        description="Key market signals and demand indicators.",
    )
    trends: list[str] = Field(
        default_factory=list,
        description="Industry trends relevant to the business.",
    )
    audience_insights: str = Field(
        default="",
        description="Summary of target audience behaviour and preferences.",
    )
    pricing_data: str = Field(
        default="",
        description="Market pricing benchmarks and observations.",
    )
    raw_search_results: list[str] = Field(
        default_factory=list,
        description="Raw search snippets used as source material.",
    )
    research_summary: str = Field(
        default="",
        description="High-level executive summary of research findings.",
    )
    is_mock: bool = Field(
        default=False,
        description="True if data came from mock fallback.",
    )


# ---------------------------------------------------------------------------
# Strategy Agent Output
# ---------------------------------------------------------------------------

class GrowthExperiment(BaseModel):
    """A single growth experiment or initiative."""
    name: str
    description: str
    expected_outcome: str = ""
    effort: str = "medium"   # low / medium / high
    priority: str = "medium" # low / medium / high


class StrategyOutput(BaseModel):
    """Structured output from the Strategy Agent."""

    gtm_strategy: str = Field(
        default="",
        description="Go-to-market strategy narrative.",
    )
    pricing_strategy: str = Field(
        default="",
        description="Recommended pricing model and rationale.",
    )
    growth_experiments: list[GrowthExperiment] = Field(
        default_factory=list,
        description="Prioritized list of growth experiments.",
    )
    market_positioning: str = Field(
        default="",
        description="Positioning statement vs. competitors.",
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Ranked strategic recommendations.",
    )
    target_segment: str = Field(
        default="",
        description="Refined target segment after analysis.",
    )
    is_mock: bool = Field(default=False)


# ---------------------------------------------------------------------------
# Planner Agent Output
# ---------------------------------------------------------------------------

class Task(BaseModel):
    """An actionable task in the execution plan."""
    name: str
    description: str = ""
    owner: str = "Team"
    duration: str = ""
    priority: str = "medium"
    dependencies: list[str] = Field(default_factory=list)


class Sprint(BaseModel):
    """A time-boxed sprint containing tasks."""
    sprint_number: int
    name: str
    duration: str = "2 weeks"
    tasks: list[str] = Field(default_factory=list)
    goal: str = ""


class Milestone(BaseModel):
    """A project milestone."""
    name: str
    target_date: str = ""
    success_criteria: str = ""


class ExecutionPlan(BaseModel):
    """Structured output from the Planner Agent."""

    tasks: list[Task] = Field(
        default_factory=list,
        description="Ordered list of actionable tasks.",
    )
    timeline: str = Field(
        default="",
        description="Overall project timeline summary.",
    )
    kpis: list[str] = Field(
        default_factory=list,
        description="Key performance indicators to track.",
    )
    sprints: list[Sprint] = Field(
        default_factory=list,
        description="Sprint breakdown.",
    )
    milestones: list[Milestone] = Field(
        default_factory=list,
        description="Key project milestones.",
    )
    implementation_steps: list[str] = Field(
        default_factory=list,
        description="High-level sequential implementation steps.",
    )
    is_mock: bool = Field(default=False)


# ---------------------------------------------------------------------------
# Critic Agent Output
# ---------------------------------------------------------------------------

class CritiqueSeverity(str, Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


class CritiqueIssue(BaseModel):
    """A single identified issue."""
    section: str
    description: str
    severity: CritiqueSeverity = CritiqueSeverity.MEDIUM
    suggestion: str = ""


class CritiqueOutput(BaseModel):
    """Structured output from the Critic Agent."""

    issues: list[CritiqueIssue] = Field(
        default_factory=list,
        description="List of identified issues across all outputs.",
    )
    overall_quality_score: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Overall quality score from 0.0 to 1.0.",
    )
    strengths: list[str] = Field(
        default_factory=list,
        description="Notable strengths identified in the outputs.",
    )
    critical_gaps: list[str] = Field(
        default_factory=list,
        description="Critical gaps that must be addressed.",
    )
    improved_summary: str = Field(
        default="",
        description="Improved executive summary suggested by critic.",
    )
    blocks_workflow: bool = Field(
        default=False,
        description="If True, workflow should pause for human review.",
    )
    is_mock: bool = Field(default=False)


# ---------------------------------------------------------------------------
# QA Agent Output
# ---------------------------------------------------------------------------

class SectionScore(BaseModel):
    """Quality score for an individual report section."""
    section: str
    score: float = Field(ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)
    passed: bool = True


class QAReport(BaseModel):
    """Structured output from the QA Agent."""

    completeness_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall completeness score from 0.0 to 1.0.",
    )
    section_scores: list[SectionScore] = Field(
        default_factory=list,
        description="Per-section quality scores.",
    )
    formatting_issues: list[str] = Field(
        default_factory=list,
        description="Detected formatting or structural problems.",
    )
    missing_sections: list[str] = Field(
        default_factory=list,
        description="Required sections that are absent or empty.",
    )
    overall_verdict: str = Field(
        default="",
        description="PASS / FAIL / PASS_WITH_WARNINGS.",
    )
    qa_summary: str = Field(
        default="",
        description="Human-readable QA summary paragraph.",
    )
    is_mock: bool = Field(default=False)

    @model_validator(mode="after")
    def derive_verdict(self) -> "QAReport":
        """Auto-set verdict if not explicitly provided."""
        if not self.overall_verdict:
            if self.completeness_score >= 0.85:
                self.overall_verdict = "PASS"
            elif self.completeness_score >= 0.6:
                self.overall_verdict = "PASS_WITH_WARNINGS"
            else:
                self.overall_verdict = "FAIL"
        return self


# ---------------------------------------------------------------------------
# Memory Schemas
# ---------------------------------------------------------------------------

class MemoryEntry(BaseModel):
    """A single entry stored in the ChromaDB vector store."""

    task_id: str
    agent_name: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)
    entry_id: Optional[str] = None


class MemoryRetrievalResult(BaseModel):
    """Result of a semantic retrieval from ChromaDB."""

    query: str
    entries: list[MemoryEntry] = Field(default_factory=list)
    total_found: int = 0
    is_summarized: bool = False
    summary: str = ""


# ---------------------------------------------------------------------------
# Observability Schemas
# ---------------------------------------------------------------------------

class ExecutionTrace(BaseModel):
    """Per-agent execution observability record."""

    agent_name: str
    action: str
    status: AgentStatus = AgentStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    model_used: str = ""
    fallback_used: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    error_message: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def compute_duration(self) -> "ExecutionTrace":
        if self.start_time and self.end_time and not self.duration_ms:
            self.duration_ms = round((self.end_time - self.start_time) * 1000, 2)
        return self


# ---------------------------------------------------------------------------
# Workflow State
# ---------------------------------------------------------------------------

class AgentState(BaseModel):
    """State of a single agent within the workflow."""

    agent_name: str
    status: AgentStatus = AgentStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    model_used: str = ""
    error_message: Optional[str] = None


class WorkflowState(BaseModel):
    """Full workflow tracking object — the source of truth for task status."""

    task_id: str
    stage: WorkflowStage = WorkflowStage.INITIALIZING
    agents: dict[str, AgentState] = Field(default_factory=dict)
    traces: list[ExecutionTrace] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    error_log: list[str] = Field(default_factory=list)
    awaiting_human_approval: bool = False

    def mark_agent_start(self, agent_name: str, model: str = "") -> None:
        self.agents[agent_name] = AgentState(
            agent_name=agent_name,
            status=AgentStatus.RUNNING,
            start_time=time.time(),
            model_used=model,
        )
        self.updated_at = time.time()

    def mark_agent_done(
        self,
        agent_name: str,
        status: AgentStatus = AgentStatus.SUCCESS,
        error: Optional[str] = None,
    ) -> None:
        state = self.agents.get(agent_name)
        if state:
            state.status = status
            state.end_time = time.time()
            if state.start_time:
                state.duration_ms = round(
                    (state.end_time - state.start_time) * 1000, 2
                )
            if error:
                state.error_message = error
                self.error_log.append(f"[{agent_name}] {error}")
        self.updated_at = time.time()

    def add_tokens(self, input_t: int, output_t: int) -> None:
        self.total_input_tokens += input_t
        self.total_output_tokens += output_t
        self.updated_at = time.time()

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def completed_agents(self) -> list[str]:
        return [
            name for name, s in self.agents.items()
            if s.status == AgentStatus.SUCCESS
        ]

    @property
    def failed_agents(self) -> list[str]:
        return [
            name for name, s in self.agents.items()
            if s.status == AgentStatus.FAILED
        ]

    @property
    def progress_percent(self) -> int:
        """Rough progress based on COMPLETED stage count."""
        stage_order = list(WorkflowStage)
        try:
            current_idx = stage_order.index(self.stage)
            return int((current_idx / (len(stage_order) - 1)) * 100)
        except ValueError:
            return 0


# ---------------------------------------------------------------------------
# Final Workflow Result
# ---------------------------------------------------------------------------

class WorkflowResult(BaseModel):
    """Consolidated final report returned by the orchestrator."""

    task_id: str
    workflow_input: Optional[WorkflowInput] = None
    research: Optional[ResearchFindings] = None
    strategy: Optional[StrategyOutput] = None
    execution_plan: Optional[ExecutionPlan] = None
    critique: Optional[CritiqueOutput] = None
    qa_report: Optional[QAReport] = None
    workflow_state: Optional[WorkflowState] = None
    generated_at: float = Field(default_factory=time.time)
    success: bool = True
    summary: str = ""
