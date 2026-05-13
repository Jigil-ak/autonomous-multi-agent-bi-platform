"""
Critic Agent
=============
Reviews all agent outputs for hallucinations, weak logic, and inconsistencies.
Never blocks workflow — always returns a critique and allows execution to continue.

Model: gemini-2.5-pro (primary), gemini-2.5-flash (fallback)
"""

from __future__ import annotations

import json
import time
from typing import Optional

from agents.schemas import (
    CritiqueOutput, CritiqueIssue, CritiqueSeverity,
    ResearchFindings, StrategyOutput, ExecutionPlan,
    WorkflowInput, WorkflowState, AgentStatus,
)
from agents.memory_agent import get_memory_agent
from utils.logger import get_logger, log_execution, estimate_tokens
from utils.model_router import call_model

logger = get_logger("critic_agent")

_SYSTEM_PROMPT = """You are a rigorous business strategy critic and quality reviewer.
Identify weaknesses, inconsistencies, and hallucinations in AI-generated business analysis.

Rules:
- Be specific and actionable in your critique
- Do not block workflow — provide constructive feedback
- Score quality from 0.0 to 1.0
- Return valid JSON only
"""

_SCHEMA_INSTRUCTION = """
Return a single valid JSON object:
{
  "issues": [
    {"section":"string","description":"string","severity":"low|medium|high","suggestion":"string"}
  ],
  "overall_quality_score": 0.0,
  "strengths": ["string"],
  "critical_gaps": ["string"],
  "improved_summary": "string",
  "blocks_workflow": false
}
Return ONLY the JSON. No markdown, no code fences.
"""


class CriticAgent:
    AGENT_NAME = "critic_agent"
    ROLE = "critic"

    def __init__(self) -> None:
        self._memory = get_memory_agent()

    def run(
        self,
        workflow_input: WorkflowInput,
        research: ResearchFindings,
        strategy: StrategyOutput,
        plan: ExecutionPlan,
        workflow_state: Optional[WorkflowState] = None,
    ) -> CritiqueOutput:
        start_time = time.time()
        task_id = workflow_input.task_id or "no-task"

        if workflow_state:
            workflow_state.mark_agent_start(self.AGENT_NAME, "gemini-2.5-pro")

        logger.info(f"[{self.AGENT_NAME}] Critiquing outputs for task={task_id}")

        try:
            prompt = f"""Critically review this AI-generated business analysis:

=== COMPANY CONTEXT ===
Company: {workflow_input.company_description}
Product: {workflow_input.product_details}
Goals: {workflow_input.goals}

=== RESEARCH FINDINGS ===
Summary: {research.research_summary[:400]}
Competitors: {', '.join(c.name for c in research.competitors[:5])}
Market Signals: {'; '.join(research.market_signals[:5])}
Trends: {'; '.join(research.trends[:4])}
Is Mock Data: {research.is_mock}

=== STRATEGY ===
GTM: {strategy.gtm_strategy[:400]}
Pricing: {strategy.pricing_strategy[:200]}
Positioning: {strategy.market_positioning[:200]}
Recommendations: {'; '.join(strategy.recommendations[:4])}
Is Mock Data: {strategy.is_mock}

=== EXECUTION PLAN ===
Timeline: {plan.timeline}
Tasks ({len(plan.tasks)}): {', '.join(t.name for t in plan.tasks[:5])}
KPIs: {'; '.join(plan.kpis[:4])}
Sprints: {len(plan.sprints)}
Is Mock Data: {plan.is_mock}

Identify:
1. Hallucinations or unsupported claims
2. Logical inconsistencies between sections
3. Missing critical business considerations
4. Weak or vague recommendations
5. Unrealistic timelines or KPIs

{_SCHEMA_INSTRUCTION}
"""
            raw_output = call_model(role=self.ROLE, prompt=prompt, system_prompt=_SYSTEM_PROMPT, temperature=0.2)
            result = self._parse_output(raw_output)

            # Critic never blocks by default (per spec)
            result.blocks_workflow = False

            self._memory.store(
                task_id=task_id, agent_name=self.AGENT_NAME,
                content=(
                    f"Quality Score: {result.overall_quality_score}\n"
                    f"Issues found: {len(result.issues)}\n"
                    f"Critical gaps: {'; '.join(result.critical_gaps[:3])}\n"
                    f"Strengths: {'; '.join(result.strengths[:3])}"
                ),
            )

            end_time = time.time()
            if workflow_state:
                workflow_state.mark_agent_done(self.AGENT_NAME, AgentStatus.SUCCESS)
                workflow_state.add_tokens(estimate_tokens(prompt), estimate_tokens(raw_output))

            log_execution(
                agent_name=self.AGENT_NAME, action="critique",
                start_time=start_time, end_time=end_time, status="success",
                metadata={"issues": len(result.issues), "score": result.overall_quality_score},
            )
            logger.info(
                f"[{self.AGENT_NAME}] Critique complete | "
                f"score={result.overall_quality_score} | issues={len(result.issues)}"
            )
            return result

        except Exception as exc:
            end_time = time.time()
            logger.error(f"[{self.AGENT_NAME}] Failed: {exc}")
            if workflow_state:
                workflow_state.mark_agent_done(self.AGENT_NAME, AgentStatus.FAILED, error=str(exc))
            log_execution(agent_name=self.AGENT_NAME, action="critique",
                          start_time=start_time, end_time=end_time, status="failed")
            return self._mock_output()

    def _parse_output(self, raw: str) -> CritiqueOutput:
        try:
            cleaned = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data = json.loads(cleaned)
            issues = [
                CritiqueIssue(
                    section=i.get("section", "General"),
                    description=i.get("description", ""),
                    severity=CritiqueSeverity(i.get("severity", "medium")),
                    suggestion=i.get("suggestion", ""),
                )
                for i in data.get("issues", []) if isinstance(i, dict)
            ]
            return CritiqueOutput(
                issues=issues,
                overall_quality_score=float(data.get("overall_quality_score", 0.7)),
                strengths=data.get("strengths", []),
                critical_gaps=data.get("critical_gaps", []),
                improved_summary=data.get("improved_summary", ""),
                blocks_workflow=False,
                is_mock=False,
            )
        except Exception as exc:
            logger.warning(f"[{self.AGENT_NAME}] Parse failed: {exc}. Using mock.")
            return self._mock_output()

    def _mock_output(self) -> CritiqueOutput:
        return CritiqueOutput(
            issues=[
                CritiqueIssue(section="Strategy", description="GTM strategy lacks specific CAC/LTV targets",
                               severity=CritiqueSeverity.MEDIUM, suggestion="Add customer acquisition cost targets"),
                CritiqueIssue(section="Execution Plan", description="Timeline may be aggressive without dedicated team",
                               severity=CritiqueSeverity.LOW, suggestion="Add risk buffer of 20% to each sprint"),
            ],
            overall_quality_score=0.78,
            strengths=["Clear market positioning", "Realistic freemium pricing", "Actionable growth experiments"],
            critical_gaps=["No competitive moat analysis", "Missing churn reduction strategy"],
            improved_summary="Solid GTM foundation with PLG focus. Strengthen with CAC/LTV analysis and retention strategy.",
            blocks_workflow=False,
            is_mock=True,
        )


def run_critic(
    workflow_input: WorkflowInput,
    research: ResearchFindings,
    strategy: StrategyOutput,
    plan: ExecutionPlan,
    workflow_state: Optional[WorkflowState] = None,
) -> CritiqueOutput:
    return CriticAgent().run(workflow_input, research, strategy, plan, workflow_state)
