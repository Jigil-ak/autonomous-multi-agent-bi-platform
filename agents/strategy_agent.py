"""
Strategy Agent
===============
Generates GTM strategy, pricing strategy, market positioning,
growth experiments, and strategic recommendations.

Model: gemini-2.5-pro (primary), gemini-2.5-flash (fallback)
"""

from __future__ import annotations

import json
import time
from typing import Optional

from agents.schemas import (
    StrategyOutput, GrowthExperiment,
    ResearchFindings, WorkflowInput, WorkflowState, AgentStatus,
)
from agents.memory_agent import get_memory_agent
from utils.logger import get_logger, log_execution, estimate_tokens
from utils.model_router import call_model

logger = get_logger("strategy_agent")

_SYSTEM_PROMPT = """You are a senior business strategist and GTM specialist with 15+ years of experience.
You generate data-driven, actionable strategic recommendations based on competitive research.

Rules:
- Ground recommendations in the provided research data
- Be specific, not generic
- Return valid JSON only — no markdown, no explanation
- All fields must be present in the output
"""

_SCHEMA_INSTRUCTION = """
Return a single valid JSON object with exactly these keys:
{
  "gtm_strategy": "string (2-4 paragraphs)",
  "pricing_strategy": "string (1-2 paragraphs with specific price points)",
  "growth_experiments": [
    {
      "name": "string",
      "description": "string",
      "expected_outcome": "string",
      "effort": "low|medium|high",
      "priority": "low|medium|high"
    }
  ],
  "market_positioning": "string (1-2 paragraphs)",
  "recommendations": ["string (ranked list of 5-7 recommendations)"],
  "target_segment": "string"
}

Return ONLY the JSON object. No markdown, no code fences.
"""


class StrategyAgent:
    AGENT_NAME = "strategy_agent"
    ROLE = "strategy"

    def __init__(self) -> None:
        self._memory = get_memory_agent()

    def run(
        self,
        workflow_input: WorkflowInput,
        research: ResearchFindings,
        workflow_state: Optional[WorkflowState] = None,
    ) -> StrategyOutput:
        start_time = time.time()
        task_id = workflow_input.task_id or "no-task"

        if workflow_state:
            workflow_state.mark_agent_start(self.AGENT_NAME, "gemini-2.5-pro")

        logger.info(f"[{self.AGENT_NAME}] Generating strategy for task={task_id}")

        try:
            # Pull relevant memory context
            context = self._memory.get_context_for_prompt(
                task_id, "market research competitors strategy"
            )

            prompt = self._build_prompt(workflow_input, research, context)
            raw_output = call_model(
                role=self.ROLE,
                prompt=prompt,
                system_prompt=_SYSTEM_PROMPT,
                temperature=0.3,
            )

            result = self._parse_output(raw_output)

            # Store in memory
            self._memory.store(
                task_id=task_id,
                agent_name=self.AGENT_NAME,
                content=(
                    f"GTM Strategy: {result.gtm_strategy[:300]}\n"
                    f"Pricing: {result.pricing_strategy[:200]}\n"
                    f"Positioning: {result.market_positioning[:200]}\n"
                    f"Recommendations: {'; '.join(result.recommendations[:3])}"
                ),
            )

            end_time = time.time()
            if workflow_state:
                workflow_state.mark_agent_done(self.AGENT_NAME, AgentStatus.SUCCESS)
                workflow_state.add_tokens(
                    estimate_tokens(prompt), estimate_tokens(raw_output)
                )

            log_execution(
                agent_name=self.AGENT_NAME, action="generate_strategy",
                start_time=start_time, end_time=end_time, status="success",
                metadata={"task_id": task_id, "experiments": len(result.growth_experiments)},
            )
            logger.info(f"[{self.AGENT_NAME}] Strategy generated successfully.")
            return result

        except Exception as exc:
            end_time = time.time()
            error_msg = f"Strategy agent failed: {exc}"
            logger.error(f"[{self.AGENT_NAME}] {error_msg}")
            if workflow_state:
                workflow_state.mark_agent_done(self.AGENT_NAME, AgentStatus.FAILED, error=error_msg)
            log_execution(
                agent_name=self.AGENT_NAME, action="generate_strategy",
                start_time=start_time, end_time=end_time, status="failed",
                metadata={"error": str(exc)},
            )
            return self._mock_output()

    def _build_prompt(
        self,
        wi: WorkflowInput,
        research: ResearchFindings,
        context: str,
    ) -> str:
        competitors_text = "\n".join(
            f"- {c.name}: {c.description} | Pricing: {c.pricing}"
            for c in research.competitors
        )
        return f"""Generate a comprehensive business strategy for:

Company: {wi.company_description}
Product: {wi.product_details}
Target Audience: {wi.target_audience}
Goals: {wi.goals}
Constraints: {wi.constraints}

=== Research Findings ===
Competitors:
{competitors_text}

Market Signals: {'; '.join(research.market_signals)}
Trends: {'; '.join(research.trends)}
Audience Insights: {research.audience_insights}
Pricing Benchmarks: {research.pricing_data}
Research Summary: {research.research_summary}

{context}

{_SCHEMA_INSTRUCTION}
"""

    def _parse_output(self, raw: str) -> StrategyOutput:
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                cleaned = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

            data = json.loads(cleaned)

            experiments = [
                GrowthExperiment(
                    name=e.get("name", "Experiment"),
                    description=e.get("description", ""),
                    expected_outcome=e.get("expected_outcome", ""),
                    effort=e.get("effort", "medium"),
                    priority=e.get("priority", "medium"),
                )
                for e in data.get("growth_experiments", [])
                if isinstance(e, dict)
            ]

            return StrategyOutput(
                gtm_strategy=data.get("gtm_strategy", ""),
                pricing_strategy=data.get("pricing_strategy", ""),
                growth_experiments=experiments,
                market_positioning=data.get("market_positioning", ""),
                recommendations=data.get("recommendations", []),
                target_segment=data.get("target_segment", ""),
                is_mock=False,
            )
        except Exception as exc:
            logger.warning(f"[{self.AGENT_NAME}] Failed to parse output: {exc}. Using mock.")
            return self._mock_output()

    def _mock_output(self) -> StrategyOutput:
        return StrategyOutput(
            gtm_strategy=(
                "Launch with a product-led growth model targeting SMBs in the analytics space. "
                "Focus on a frictionless free tier that lets users experience value before paying."
            ),
            pricing_strategy=(
                "Freemium: $0 for up to 3 users and 5 dashboards. "
                "Growth tier: $29/user/month. Enterprise: custom pricing."
            ),
            growth_experiments=[
                GrowthExperiment(
                    name="Viral referral program",
                    description="Reward users with extended free tier for successful referrals.",
                    expected_outcome="30% increase in organic signups",
                    effort="low", priority="high",
                ),
                GrowthExperiment(
                    name="Content-led SEO",
                    description="Publish weekly BI best practices and comparison guides.",
                    expected_outcome="10k organic monthly visitors in 6 months",
                    effort="medium", priority="high",
                ),
            ],
            market_positioning=(
                "The simplest AI-powered BI tool built for growing teams. "
                "No SQL required. No data engineer needed."
            ),
            recommendations=[
                "Prioritize frictionless onboarding — first value in under 5 minutes",
                "Build integrations with top 5 SMB tools (Notion, HubSpot, Airtable)",
                "Invest in bottom-up PLG before enterprise sales",
                "Target industries with high data complexity: e-commerce, SaaS, agencies",
                "Publish transparent pricing to build trust with SMBs",
            ],
            target_segment="SMBs with 10-200 employees in data-intensive industries",
            is_mock=True,
        )


def run_strategy(
    workflow_input: WorkflowInput,
    research: ResearchFindings,
    workflow_state: Optional[WorkflowState] = None,
) -> StrategyOutput:
    return StrategyAgent().run(workflow_input, research, workflow_state)
