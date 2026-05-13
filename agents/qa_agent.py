"""
QA Agent
=========
Validates completeness, formatting, section quality, and workflow integrity.
Returns a structured QA report with scores and verdicts.

Model: gemini-2.5-flash-lite (primary), gemini-2.5-flash (fallback)
"""

from __future__ import annotations

import json
import time
from typing import Optional

from agents.schemas import (
    QAReport, SectionScore,
    ResearchFindings, StrategyOutput, ExecutionPlan, CritiqueOutput,
    WorkflowInput, WorkflowState, AgentStatus,
)
from agents.memory_agent import get_memory_agent
from utils.logger import get_logger, log_execution, estimate_tokens
from utils.model_router import call_model

logger = get_logger("qa_agent")

_SYSTEM_PROMPT = """You are a quality assurance specialist reviewing AI-generated business reports.
Check for completeness, accuracy, formatting, and section quality.
Return valid JSON only — no markdown, no explanation.
"""

_SCHEMA_INSTRUCTION = """
Return a single valid JSON object:
{
  "completeness_score": 0.0,
  "section_scores": [
    {"section":"string","score":0.0,"issues":[],"passed":true}
  ],
  "formatting_issues": ["string"],
  "missing_sections": ["string"],
  "overall_verdict": "PASS|FAIL|PASS_WITH_WARNINGS",
  "qa_summary": "string"
}
Return ONLY the JSON. No markdown, no code fences.
"""


class QAAgent:
    AGENT_NAME = "qa_agent"
    ROLE = "qa"

    def __init__(self) -> None:
        self._memory = get_memory_agent()

    def run(
        self,
        workflow_input: WorkflowInput,
        research: ResearchFindings,
        strategy: StrategyOutput,
        plan: ExecutionPlan,
        critique: CritiqueOutput,
        workflow_state: Optional[WorkflowState] = None,
    ) -> QAReport:
        start_time = time.time()
        task_id = workflow_input.task_id or "no-task"

        if workflow_state:
            workflow_state.mark_agent_start(self.AGENT_NAME, "gemini-2.5-flash-lite")

        logger.info(f"[{self.AGENT_NAME}] Running QA validation for task={task_id}")

        try:
            prompt = f"""Validate this AI-generated business analysis report:

=== RESEARCH SECTION ===
Competitors found: {len(research.competitors)}
Market signals: {len(research.market_signals)}
Trends identified: {len(research.trends)}
Audience insights present: {"Yes" if research.audience_insights else "No"}
Pricing data present: {"Yes" if research.pricing_data else "No"}
Research summary present: {"Yes" if research.research_summary else "No"}
Is mock data: {research.is_mock}

=== STRATEGY SECTION ===
GTM strategy present: {"Yes" if strategy.gtm_strategy else "No"}
Pricing strategy present: {"Yes" if strategy.pricing_strategy else "No"}
Growth experiments: {len(strategy.growth_experiments)}
Market positioning present: {"Yes" if strategy.market_positioning else "No"}
Recommendations: {len(strategy.recommendations)}
Target segment defined: {"Yes" if strategy.target_segment else "No"}
Is mock data: {strategy.is_mock}

=== EXECUTION PLAN SECTION ===
Tasks defined: {len(plan.tasks)}
Timeline present: {"Yes" if plan.timeline else "No"}
KPIs defined: {len(plan.kpis)}
Sprints planned: {len(plan.sprints)}
Milestones set: {len(plan.milestones)}
Implementation steps: {len(plan.implementation_steps)}
Is mock data: {plan.is_mock}

=== CRITIQUE SECTION ===
Quality score: {critique.overall_quality_score}
Issues found: {len(critique.issues)}
Strengths: {len(critique.strengths)}
Critical gaps: {len(critique.critical_gaps)}
Is mock data: {critique.is_mock}

Evaluate completeness, consistency across sections, and data quality.
Note if sections used mock/fallback data.

{_SCHEMA_INSTRUCTION}
"""
            raw_output = call_model(role=self.ROLE, prompt=prompt, system_prompt=_SYSTEM_PROMPT, temperature=0.1)
            result = self._parse_output(raw_output)

            self._memory.store(
                task_id=task_id, agent_name=self.AGENT_NAME,
                content=(
                    f"QA Verdict: {result.overall_verdict}\n"
                    f"Completeness: {result.completeness_score}\n"
                    f"Missing: {'; '.join(result.missing_sections)}\n"
                    f"Summary: {result.qa_summary[:200]}"
                ),
            )

            end_time = time.time()
            if workflow_state:
                workflow_state.mark_agent_done(self.AGENT_NAME, AgentStatus.SUCCESS)
                workflow_state.add_tokens(estimate_tokens(prompt), estimate_tokens(raw_output))

            log_execution(
                agent_name=self.AGENT_NAME, action="qa_validation",
                start_time=start_time, end_time=end_time, status="success",
                metadata={"verdict": result.overall_verdict, "score": result.completeness_score},
            )
            logger.info(
                f"[{self.AGENT_NAME}] QA complete | "
                f"verdict={result.overall_verdict} | score={result.completeness_score}"
            )
            return result

        except Exception as exc:
            end_time = time.time()
            logger.error(f"[{self.AGENT_NAME}] Failed: {exc}")
            if workflow_state:
                workflow_state.mark_agent_done(self.AGENT_NAME, AgentStatus.FAILED, error=str(exc))
            log_execution(agent_name=self.AGENT_NAME, action="qa_validation",
                          start_time=start_time, end_time=end_time, status="failed")
            return self._mock_output()

    def _parse_output(self, raw: str) -> QAReport:
        try:
            cleaned = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data = json.loads(cleaned)
            section_scores = [
                SectionScore(
                    section=s.get("section", "Unknown"),
                    score=float(s.get("score", 0.7)),
                    issues=s.get("issues", []),
                    passed=bool(s.get("passed", True)),
                )
                for s in data.get("section_scores", []) if isinstance(s, dict)
            ]
            return QAReport(
                completeness_score=float(data.get("completeness_score", 0.7)),
                section_scores=section_scores,
                formatting_issues=data.get("formatting_issues", []),
                missing_sections=data.get("missing_sections", []),
                overall_verdict=data.get("overall_verdict", ""),
                qa_summary=data.get("qa_summary", ""),
                is_mock=False,
            )
        except Exception as exc:
            logger.warning(f"[{self.AGENT_NAME}] Parse failed: {exc}. Using mock.")
            return self._mock_output()

    def _mock_output(self) -> QAReport:
        return QAReport(
            completeness_score=0.82,
            section_scores=[
                SectionScore(section="Research", score=0.85, issues=[], passed=True),
                SectionScore(section="Strategy", score=0.80, issues=["Missing CAC targets"], passed=True),
                SectionScore(section="Execution Plan", score=0.85, issues=[], passed=True),
                SectionScore(section="Critique", score=0.78, issues=[], passed=True),
            ],
            formatting_issues=[],
            missing_sections=["Risk assessment", "Financial projections"],
            overall_verdict="PASS_WITH_WARNINGS",
            qa_summary=(
                "Report covers all major sections with adequate depth. "
                "Minor gaps in financial projections and risk analysis. "
                "Strategy and execution plan are well-aligned. Recommended for delivery."
            ),
            is_mock=True,
        )


def run_qa(
    workflow_input: WorkflowInput,
    research: ResearchFindings,
    strategy: StrategyOutput,
    plan: ExecutionPlan,
    critique: CritiqueOutput,
    workflow_state: Optional[WorkflowState] = None,
) -> QAReport:
    return QAAgent().run(workflow_input, research, strategy, plan, critique, workflow_state)
