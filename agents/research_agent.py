"""
Research Agent
===============
Performs competitor analysis, market signal extraction, trend identification,
and audience research using DuckDuckGo search + Gemini LLM synthesis.

Model: gemini-2.5-flash (primary), gemini-2.5-flash-lite (fallback)

Workflow:
1. Build targeted search queries from workflow input
2. Execute searches (rate-limited, mock-resilient)
3. Format raw results into a structured LLM prompt
4. Call LLM to synthesize findings into ResearchFindings schema
5. Parse + validate output against Pydantic schema
6. Store results in ChromaDB memory
7. Return ResearchFindings
"""

from __future__ import annotations

import json
import time
from typing import Optional

from agents.schemas import ResearchFindings, Competitor, WorkflowInput, WorkflowState, AgentStatus
from memory.chroma_store import get_memory_store
from tools.search_tools import (
    run_research_queries,
    format_results_as_text,
    search_competitors,
    search_market_trends,
    search_pricing,
    search_audience_insights,
)
from utils.logger import get_logger, log_execution, estimate_tokens
from utils.model_router import call_model

logger = get_logger("research_agent")

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are a professional market research analyst AI.
Your role is to analyze search results and extract structured competitive intelligence.

Rules:
- Extract factual information only from the provided search results
- Do not fabricate data not present in the search snippets
- Always return valid JSON matching the exact schema specified
- Be concise and specific
"""

# ---------------------------------------------------------------------------
# Output schema instruction for the LLM
# ---------------------------------------------------------------------------
_SCHEMA_INSTRUCTION = """
Return your analysis as a single valid JSON object with exactly these keys:
{
  "competitors": [
    {
      "name": "string",
      "description": "string",
      "strengths": ["string"],
      "weaknesses": ["string"],
      "pricing": "string"
    }
  ],
  "market_signals": ["string"],
  "trends": ["string"],
  "audience_insights": "string",
  "pricing_data": "string",
  "research_summary": "string"
}

Return ONLY the JSON object. No markdown, no explanation, no code fences.
"""


class ResearchAgent:
    """
    Research Agent — collects and synthesizes competitive intelligence.
    """

    AGENT_NAME = "research_agent"
    ROLE = "research"

    def __init__(self) -> None:
        self._memory = get_memory_store()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        workflow_input: WorkflowInput,
        workflow_state: Optional[WorkflowState] = None,
    ) -> ResearchFindings:
        """
        Execute the research pipeline.

        Parameters
        ----------
        workflow_input : WorkflowInput
            Business context provided by the user.
        workflow_state : WorkflowState, optional
            Shared workflow state for progress tracking.

        Returns
        -------
        ResearchFindings
            Structured competitive intelligence.
        """
        start_time = time.time()
        task_id = workflow_input.task_id or "no-task"

        if workflow_state:
            workflow_state.mark_agent_start(self.AGENT_NAME, "gemini-2.5-flash")

        logger.info(
            f"[{self.AGENT_NAME}] Starting research for task={task_id} | "
            f"company='{workflow_input.company_description[:40]}'"
        )

        try:
            # Step 1: Build queries
            queries = self._build_queries(workflow_input)

            # Step 2: Execute searches
            raw_results = self._execute_searches(workflow_input, queries)

            # Step 3: Synthesize with LLM
            findings = self._synthesize(workflow_input, raw_results)

            # Step 4: Store in memory
            self._store_results(task_id, findings)

            # Step 5: Update workflow state
            end_time = time.time()
            if workflow_state:
                workflow_state.mark_agent_done(self.AGENT_NAME, AgentStatus.SUCCESS)
                workflow_state.add_tokens(
                    estimate_tokens(str(workflow_input.model_dump())),
                    estimate_tokens(str(findings.model_dump())),
                )

            log_execution(
                agent_name=self.AGENT_NAME,
                action="research_pipeline",
                start_time=start_time,
                end_time=end_time,
                status="success",
                metadata={"task_id": task_id, "competitors_found": len(findings.competitors)},
            )

            logger.info(
                f"[{self.AGENT_NAME}] Research complete | "
                f"competitors={len(findings.competitors)} | "
                f"signals={len(findings.market_signals)}"
            )
            return findings

        except Exception as exc:
            end_time = time.time()
            error_msg = f"Research agent failed: {exc}"
            logger.error(f"[{self.AGENT_NAME}] {error_msg}")

            if workflow_state:
                workflow_state.mark_agent_done(
                    self.AGENT_NAME, AgentStatus.FAILED, error=error_msg
                )

            log_execution(
                agent_name=self.AGENT_NAME,
                action="research_pipeline",
                start_time=start_time,
                end_time=end_time,
                status="failed",
                metadata={"error": str(exc)},
            )

            # Return mock findings so workflow can continue
            return self._mock_findings()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_queries(self, wi: WorkflowInput) -> list[str]:
        """Build targeted search queries from workflow input."""
        # Extract key terms
        company = wi.company_description[:60]
        product = wi.product_details[:60]
        audience = wi.target_audience[:40]

        queries = [
            f"{product} competitors alternatives 2024 2025",
            f"{product} market trends growth {audience}",
            f"{product} pricing comparison cost per user",
            f"{audience} {product} pain points needs",
        ]
        logger.info(f"[{self.AGENT_NAME}] Built {len(queries)} search queries.")
        return queries

    def _execute_searches(
        self, wi: WorkflowInput, queries: list[str]
    ) -> list[dict]:
        """Run all search queries with rate limiting."""
        logger.info(f"[{self.AGENT_NAME}] Executing {len(queries)} searches...")
        results = run_research_queries(queries, max_results_per_query=4)
        logger.info(f"[{self.AGENT_NAME}] Got {len(results)} total search results.")
        return results

    def _synthesize(
        self,
        wi: WorkflowInput,
        raw_results: list[dict],
    ) -> ResearchFindings:
        """Call LLM to synthesize raw search results into structured findings."""
        search_text = format_results_as_text(raw_results)

        prompt = f"""You are analyzing market research for this business:

Company: {wi.company_description}
Product: {wi.product_details}
Target Audience: {wi.target_audience}
Goals: {wi.goals}

Here are the search results collected from the web:

{search_text}

{_SCHEMA_INSTRUCTION}
"""
        logger.info(f"[{self.AGENT_NAME}] Calling LLM to synthesize research...")
        raw_output = call_model(
            role=self.ROLE,
            prompt=prompt,
            system_prompt=_SYSTEM_PROMPT,
            temperature=0.2,
        )

        return self._parse_output(raw_output, raw_results)

    def _parse_output(
        self,
        raw_output: str,
        raw_results: list[dict],
    ) -> ResearchFindings:
        """Parse and validate LLM output against ResearchFindings schema."""
        try:
            # Strip any accidental markdown fences
            cleaned = raw_output.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                cleaned = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

            data = json.loads(cleaned)

            # Normalize competitors
            competitors = []
            for c in data.get("competitors", []):
                if isinstance(c, dict):
                    competitors.append(Competitor(
                        name=c.get("name", "Unknown"),
                        description=c.get("description", ""),
                        strengths=c.get("strengths", []),
                        weaknesses=c.get("weaknesses", []),
                        pricing=c.get("pricing", ""),
                    ))
                elif isinstance(c, str):
                    competitors.append(Competitor(name=c))

            # Collect raw snippets
            raw_snippets = [
                r.get("body", "") for r in raw_results if r.get("body")
            ][:10]

            findings = ResearchFindings(
                competitors=competitors,
                market_signals=data.get("market_signals", []),
                trends=data.get("trends", []),
                audience_insights=data.get("audience_insights", ""),
                pricing_data=data.get("pricing_data", ""),
                raw_search_results=raw_snippets,
                research_summary=data.get("research_summary", ""),
                is_mock=False,
            )
            logger.info(f"[{self.AGENT_NAME}] Research findings validated successfully.")
            return findings

        except (json.JSONDecodeError, Exception) as exc:
            logger.warning(
                f"[{self.AGENT_NAME}] Failed to parse LLM output: {exc}. "
                f"Falling back to mock findings."
            )
            return self._mock_findings()

    def _store_results(self, task_id: str, findings: ResearchFindings) -> None:
        """Persist findings in ChromaDB memory."""
        content = (
            f"Research Summary: {findings.research_summary}\n\n"
            f"Competitors: {', '.join(c.name for c in findings.competitors)}\n\n"
            f"Market Signals: {'; '.join(findings.market_signals)}\n\n"
            f"Trends: {'; '.join(findings.trends)}\n\n"
            f"Audience: {findings.audience_insights}\n\n"
            f"Pricing: {findings.pricing_data}"
        )
        self._memory.store(
            task_id=task_id,
            agent_name=self.AGENT_NAME,
            content=content,
            metadata={"stage": "research", "competitor_count": str(len(findings.competitors))},
        )

    def _mock_findings(self) -> ResearchFindings:
        """Return structured mock findings for demo resilience."""
        return ResearchFindings(
            competitors=[
                Competitor(
                    name="Tableau",
                    description="Leading BI platform with rich visualization capabilities.",
                    strengths=["Market leader", "Extensive integrations", "Strong community"],
                    weaknesses=["High cost", "Complex setup", "Steep learning curve"],
                    pricing="$70-$840/user/year",
                ),
                Competitor(
                    name="Power BI",
                    description="Microsoft's BI tool deeply integrated with Office 365.",
                    strengths=["Low cost for Microsoft users", "Wide adoption", "Strong Excel integration"],
                    weaknesses=["Windows-centric", "Limited for non-Microsoft stacks"],
                    pricing="$10/user/month",
                ),
            ],
            market_signals=[
                "Growing demand for AI-powered analytics",
                "Shift towards self-service BI tools",
                "SMBs increasing analytics budgets",
            ],
            trends=[
                "Natural language querying",
                "Embedded analytics in SaaS products",
                "Real-time dashboards",
            ],
            audience_insights="SMBs prefer easy-to-use tools with fast setup and transparent pricing.",
            pricing_data="Market range: $10-$100/user/month. Freemium models gaining traction.",
            research_summary=(
                "The BI market is growing rapidly. Key competitors include Tableau and Power BI. "
                "SMB segment is underserved with complex tools. Opportunity exists for AI-native, "
                "easy-to-use platforms with transparent pricing."
            ),
            is_mock=True,
        )


# ---------------------------------------------------------------------------
# Convenience function for direct use by orchestrator
# ---------------------------------------------------------------------------
def run_research(
    workflow_input: WorkflowInput,
    workflow_state: Optional[WorkflowState] = None,
) -> ResearchFindings:
    """Create a ResearchAgent and run the research pipeline."""
    agent = ResearchAgent()
    return agent.run(workflow_input, workflow_state)
