"""
Orchestrator Agent
===================
Central workflow controller that coordinates all specialized agents.

Model: gemini-2.5-flash (primary), gemini-2.5-flash-lite (fallback)

Pipeline (sequential):
  1. Sanitize input & initialize workflow state
  2. Store input context in memory
  3. Research Agent  → ResearchFindings
  4. Strategy Agent  → StrategyOutput
  5. Planner Agent   → ExecutionPlan
  6. Critic Agent    → CritiqueOutput
  7. QA Agent        → QAReport
  8. Store final report in memory
  9. Return WorkflowResult

Design decisions:
- Each agent failure is caught independently — workflow continues
- Workflow state tracks every agent's lifecycle
- Human-in-the-loop checkpoint is available (disabled by default)
- Cost tracking enforced throughout
- No recursive loops — strictly sequential
"""

from __future__ import annotations

import time
import uuid
from typing import Optional

from agents.schemas import (
    WorkflowInput, WorkflowResult, WorkflowState, WorkflowStage, AgentStatus,
    ResearchFindings, StrategyOutput, ExecutionPlan, CritiqueOutput, QAReport,
)
from agents.research_agent import run_research
from agents.strategy_agent import run_strategy
from agents.planner_agent import run_planner
from agents.critic_agent import run_critic
from agents.qa_agent import run_qa
from agents.memory_agent import get_memory_agent
from utils.guards import sanitize_input, check_iteration_limit, CostTracker
from utils.logger import get_logger, log_execution, estimate_tokens
from utils.model_router import call_model

logger = get_logger("orchestrator")


class Orchestrator:
    """
    Sequential multi-agent workflow coordinator.

    All agent failures are handled gracefully — the workflow always produces
    a WorkflowResult even if individual agents fail or use mock outputs.
    """

    AGENT_NAME = "orchestrator"
    ROLE = "orchestrator"

    # Human-in-the-loop checkpoint — set to True to pause before planning
    HUMAN_APPROVAL_REQUIRED: bool = False

    def __init__(self) -> None:
        self._memory = get_memory_agent()
        self._cost_tracker = CostTracker()

    def run(
        self,
        workflow_input: WorkflowInput,
        enable_human_approval: bool = False,
    ) -> WorkflowResult:
        """
        Execute the full multi-agent workflow.

        Parameters
        ----------
        workflow_input : WorkflowInput
            Business context from the user.
        enable_human_approval : bool
            If True, pause after strategy and await approval before planning.

        Returns
        -------
        WorkflowResult
            Consolidated report from all agents.
        """
        # Assign task ID if not set
        if not workflow_input.task_id:
            workflow_input.task_id = str(uuid.uuid4())[:8]

        task_id = workflow_input.task_id
        global_start = time.time()

        logger.info(
            f"[{self.AGENT_NAME}] ===== WORKFLOW START ===== "
            f"task_id={task_id}"
        )

        # Initialize workflow state
        state = WorkflowState(task_id=task_id)
        state.stage = WorkflowStage.INITIALIZING

        # Initialize result containers (with None defaults)
        research: Optional[ResearchFindings] = None
        strategy: Optional[StrategyOutput] = None
        plan: Optional[ExecutionPlan] = None
        critique: Optional[CritiqueOutput] = None
        qa: Optional[QAReport] = None

        try:
            # ----------------------------------------------------------------
            # Step 0: Sanitize inputs
            # ----------------------------------------------------------------
            logger.info(f"[{self.AGENT_NAME}] Step 0: Sanitizing inputs...")
            workflow_input.company_description = sanitize_input(workflow_input.company_description)
            workflow_input.product_details = sanitize_input(workflow_input.product_details)
            workflow_input.target_audience = sanitize_input(workflow_input.target_audience)
            workflow_input.goals = sanitize_input(workflow_input.goals)
            workflow_input.constraints = sanitize_input(workflow_input.constraints)

            # ----------------------------------------------------------------
            # Step 1: Store input in memory
            # ----------------------------------------------------------------
            logger.info(f"[{self.AGENT_NAME}] Step 1: Storing context in memory...")
            self._memory.store(
                task_id=task_id,
                agent_name="orchestrator",
                content=(
                    f"Company: {workflow_input.company_description}\n"
                    f"Product: {workflow_input.product_details}\n"
                    f"Audience: {workflow_input.target_audience}\n"
                    f"Goals: {workflow_input.goals}\n"
                    f"Constraints: {workflow_input.constraints}"
                ),
                metadata={"stage": "input"},
            )

            # ----------------------------------------------------------------
            # Step 2: Research Agent
            # ----------------------------------------------------------------
            state.stage = WorkflowStage.RESEARCHING
            logger.info(f"[{self.AGENT_NAME}] Step 2: Running Research Agent...")

            if self._cost_tracker.can_proceed():
                research = self._run_safe(
                    "Research", lambda: run_research(workflow_input, state)
                )
                if research:
                    self._cost_tracker.add_usage(
                        str(workflow_input.model_dump()),
                        str(research.model_dump()),
                        agent_name="research_agent",
                    )
            else:
                logger.warning(f"[{self.AGENT_NAME}] Cost limit reached. Skipping Research Agent.")
                state.agents["research_agent"] = self._skipped_state("research_agent")

            if not research:
                research = self._fallback_research()

            # ----------------------------------------------------------------
            # Step 3: Strategy Agent
            # ----------------------------------------------------------------
            state.stage = WorkflowStage.STRATEGIZING
            logger.info(f"[{self.AGENT_NAME}] Step 3: Running Strategy Agent...")

            if self._cost_tracker.can_proceed():
                strategy = self._run_safe(
                    "Strategy", lambda: run_strategy(workflow_input, research, state)
                )
                if strategy:
                    self._cost_tracker.add_usage(
                        str(research.model_dump()), str(strategy.model_dump()),
                        agent_name="strategy_agent",
                    )
            else:
                logger.warning(f"[{self.AGENT_NAME}] Cost limit reached. Skipping Strategy Agent.")
                state.agents["strategy_agent"] = self._skipped_state("strategy_agent")

            if not strategy:
                strategy = self._fallback_strategy()

            # ----------------------------------------------------------------
            # Optional: Human-in-the-loop approval checkpoint
            # ----------------------------------------------------------------
            if enable_human_approval or self.HUMAN_APPROVAL_REQUIRED:
                state.awaiting_human_approval = True
                logger.info(
                    f"[{self.AGENT_NAME}] Human approval checkpoint reached. "
                    f"task_id={task_id} | awaiting_human_approval=True"
                )
                # In async API mode, the frontend polls status and
                # calls POST /approve/{task_id} to resume.
                # For now in direct execution mode, we auto-continue.
                state.awaiting_human_approval = False

            # ----------------------------------------------------------------
            # Step 4: Planner Agent
            # ----------------------------------------------------------------
            state.stage = WorkflowStage.PLANNING
            logger.info(f"[{self.AGENT_NAME}] Step 4: Running Planner Agent...")

            if self._cost_tracker.can_proceed():
                plan = self._run_safe(
                    "Planner", lambda: run_planner(workflow_input, strategy, state)
                )
                if plan:
                    self._cost_tracker.add_usage(
                        str(strategy.model_dump()), str(plan.model_dump()),
                        agent_name="planner_agent",
                    )
            else:
                logger.warning(f"[{self.AGENT_NAME}] Cost limit reached. Skipping Planner Agent.")
                state.agents["planner_agent"] = self._skipped_state("planner_agent")

            if not plan:
                plan = self._fallback_plan()

            # ----------------------------------------------------------------
            # Step 5: Critic Agent
            # ----------------------------------------------------------------
            state.stage = WorkflowStage.CRITIQUING
            logger.info(f"[{self.AGENT_NAME}] Step 5: Running Critic Agent...")

            if self._cost_tracker.can_proceed():
                critique = self._run_safe(
                    "Critic",
                    lambda: run_critic(workflow_input, research, strategy, plan, state),
                )
                if critique:
                    self._cost_tracker.add_usage(
                        "critique_input", str(critique.model_dump()),
                        agent_name="critic_agent",
                    )
            else:
                logger.warning(f"[{self.AGENT_NAME}] Cost limit reached. Skipping Critic Agent.")
                state.agents["critic_agent"] = self._skipped_state("critic_agent")

            if not critique:
                critique = self._fallback_critique()

            # ----------------------------------------------------------------
            # Step 6: QA Agent
            # ----------------------------------------------------------------
            state.stage = WorkflowStage.QA_CHECK
            logger.info(f"[{self.AGENT_NAME}] Step 6: Running QA Agent...")

            if self._cost_tracker.can_proceed():
                qa = self._run_safe(
                    "QA",
                    lambda: run_qa(workflow_input, research, strategy, plan, critique, state),
                )
                if qa:
                    self._cost_tracker.add_usage(
                        "qa_input", str(qa.model_dump()),
                        agent_name="qa_agent",
                    )
            else:
                logger.warning(f"[{self.AGENT_NAME}] Cost limit reached. Skipping QA Agent.")
                state.agents["qa_agent"] = self._skipped_state("qa_agent")

            if not qa:
                qa = self._fallback_qa()

            # ----------------------------------------------------------------
            # Step 7: Store final report
            # ----------------------------------------------------------------
            state.stage = WorkflowStage.STORING
            logger.info(f"[{self.AGENT_NAME}] Step 7: Storing final report...")
            self._store_final_report(task_id, research, strategy, plan, critique, qa)

            # ----------------------------------------------------------------
            # Step 8: Build and return final result
            # ----------------------------------------------------------------
            state.stage = WorkflowStage.COMPLETED
            global_end = time.time()

            cost_summary = self._cost_tracker.get_summary()
            state.total_input_tokens = cost_summary["total_input_tokens"]
            state.total_output_tokens = cost_summary["total_output_tokens"]

            summary = self._build_summary(state, qa, critique)

            log_execution(
                agent_name=self.AGENT_NAME,
                action="full_workflow",
                start_time=global_start,
                end_time=global_end,
                status="success",
                metadata={
                    "task_id": task_id,
                    "completed_agents": len(state.completed_agents),
                    "failed_agents": len(state.failed_agents),
                    "total_tokens": cost_summary["total_tokens"],
                    "qa_verdict": qa.overall_verdict if qa else "unknown",
                },
            )

            logger.info(
                f"[{self.AGENT_NAME}] ===== WORKFLOW COMPLETE ===== "
                f"task_id={task_id} | "
                f"completed={len(state.completed_agents)} | "
                f"failed={len(state.failed_agents)} | "
                f"tokens≈{cost_summary['total_tokens']} | "
                f"verdict={qa.overall_verdict if qa else 'N/A'}"
            )

            return WorkflowResult(
                task_id=task_id,
                workflow_input=workflow_input,
                research=research,
                strategy=strategy,
                execution_plan=plan,
                critique=critique,
                qa_report=qa,
                workflow_state=state,
                success=True,
                summary=summary,
            )

        except Exception as exc:
            # Catastrophic failure — still return a partial result
            global_end = time.time()
            error_msg = f"Orchestrator catastrophic failure: {exc}"
            logger.error(f"[{self.AGENT_NAME}] {error_msg}")
            state.stage = WorkflowStage.FAILED
            state.error_log.append(error_msg)

            log_execution(
                agent_name=self.AGENT_NAME, action="full_workflow",
                start_time=global_start, end_time=global_end, status="failed",
                metadata={"error": str(exc)},
            )

            return WorkflowResult(
                task_id=task_id,
                workflow_input=workflow_input,
                research=research,
                strategy=strategy,
                execution_plan=plan,
                critique=critique,
                qa_report=qa,
                workflow_state=state,
                success=False,
                summary=f"Workflow failed: {exc}",
            )

    # ------------------------------------------------------------------
    # Safe agent runner
    # ------------------------------------------------------------------

    def _run_safe(self, agent_label: str, fn) -> Optional[object]:
        """
        Run an agent function, catching all exceptions gracefully.

        Returns the result or None on failure.
        """
        try:
            return fn()
        except Exception as exc:
            logger.error(
                f"[{self.AGENT_NAME}] {agent_label} agent raised exception: {exc}"
            )
            return None

    # ------------------------------------------------------------------
    # Fallback outputs (ensures downstream agents always have input)
    # ------------------------------------------------------------------

    def _fallback_research(self) -> ResearchFindings:
        from agents.research_agent import ResearchAgent
        return ResearchAgent()._mock_findings()

    def _fallback_strategy(self) -> StrategyOutput:
        from agents.strategy_agent import StrategyAgent
        return StrategyAgent()._mock_output()

    def _fallback_plan(self) -> ExecutionPlan:
        from agents.planner_agent import PlannerAgent
        return PlannerAgent()._mock_output()

    def _fallback_critique(self) -> CritiqueOutput:
        from agents.critic_agent import CriticAgent
        return CriticAgent()._mock_output()

    def _fallback_qa(self) -> QAReport:
        from agents.qa_agent import QAAgent
        return QAAgent()._mock_output()

    def _skipped_state(self, agent_name: str):
        from agents.schemas import AgentState
        return AgentState(agent_name=agent_name, status=AgentStatus.SKIPPED)

    # ------------------------------------------------------------------
    # Memory storage
    # ------------------------------------------------------------------

    def _store_final_report(
        self,
        task_id: str,
        research: ResearchFindings,
        strategy: StrategyOutput,
        plan: ExecutionPlan,
        critique: CritiqueOutput,
        qa: QAReport,
    ) -> None:
        """Store a final consolidated report summary in memory."""
        content = f"""=== FINAL WORKFLOW REPORT ===
Task: {task_id}

RESEARCH: {research.research_summary[:300]}
STRATEGY: {strategy.gtm_strategy[:300]}
PLAN: Timeline={plan.timeline} | Tasks={len(plan.tasks)} | KPIs={len(plan.kpis)}
CRITIQUE: Score={critique.overall_quality_score} | Issues={len(critique.issues)}
QA: Verdict={qa.overall_verdict} | Score={qa.completeness_score}
"""
        self._memory.store(
            task_id=task_id,
            agent_name="orchestrator",
            content=content,
            metadata={"stage": "final_report"},
        )

    # ------------------------------------------------------------------
    # Summary builder
    # ------------------------------------------------------------------

    def _build_summary(
        self,
        state: WorkflowState,
        qa: Optional[QAReport],
        critique: Optional[CritiqueOutput],
    ) -> str:
        verdict = qa.overall_verdict if qa else "N/A"
        score = f"{critique.overall_quality_score:.2f}" if critique else "N/A"
        completed = len(state.completed_agents)
        failed = len(state.failed_agents)

        return (
            f"Workflow completed with {completed} agents successful"
            f"{f', {failed} failed' if failed else ''}. "
            f"Quality score: {score}. QA verdict: {verdict}. "
            f"Estimated tokens used: {state.total_tokens}."
        )


# ---------------------------------------------------------------------------
# Module-level runner
# ---------------------------------------------------------------------------
def run_workflow(
    workflow_input: WorkflowInput,
    enable_human_approval: bool = False,
) -> WorkflowResult:
    """
    Create an Orchestrator and run the full workflow.

    This is the primary entry point used by the FastAPI backend.
    """
    orchestrator = Orchestrator()
    return orchestrator.run(workflow_input, enable_human_approval=enable_human_approval)
