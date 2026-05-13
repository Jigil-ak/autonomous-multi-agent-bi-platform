"""
Planner Agent
==============
Converts strategic output into a concrete execution plan with tasks,
KPIs, sprints, milestones, and timelines.

Model: gemini-2.5-flash (primary), gemini-2.5-flash-lite (fallback)
"""

from __future__ import annotations

import json
import time
from typing import Optional

from agents.schemas import (
    ExecutionPlan, Task, Sprint, Milestone,
    StrategyOutput, WorkflowInput, WorkflowState, AgentStatus,
)
from agents.memory_agent import get_memory_agent
from utils.logger import get_logger, log_execution, estimate_tokens
from utils.model_router import call_model

logger = get_logger("planner_agent")

_SYSTEM_PROMPT = """You are a senior product manager and execution specialist.
Convert high-level strategy into detailed, actionable implementation plans.
Return valid JSON only — no markdown, no explanation.
"""

_SCHEMA_INSTRUCTION = """
Return a single valid JSON object:
{
  "tasks": [{"name":"","description":"","owner":"","duration":"","priority":"medium","dependencies":[]}],
  "timeline": "string",
  "kpis": ["string"],
  "sprints": [{"sprint_number":1,"name":"","duration":"2 weeks","tasks":[],"goal":""}],
  "milestones": [{"name":"","target_date":"","success_criteria":""}],
  "implementation_steps": ["string"]
}
Return ONLY the JSON. No markdown, no code fences.
"""


class PlannerAgent:
    AGENT_NAME = "planner_agent"
    ROLE = "planning"

    def __init__(self) -> None:
        self._memory = get_memory_agent()

    def run(
        self,
        workflow_input: WorkflowInput,
        strategy: StrategyOutput,
        workflow_state: Optional[WorkflowState] = None,
    ) -> ExecutionPlan:
        start_time = time.time()
        task_id = workflow_input.task_id or "no-task"

        if workflow_state:
            workflow_state.mark_agent_start(self.AGENT_NAME, "gemini-2.5-flash")

        logger.info(f"[{self.AGENT_NAME}] Building plan for task={task_id}")

        try:
            context = self._memory.get_context_for_prompt(task_id, "strategy GTM recommendations")
            recs = "\n".join(f"- {r}" for r in strategy.recommendations[:5])
            exps = "\n".join(f"- {e.name}: {e.description}" for e in strategy.growth_experiments[:3])

            prompt = f"""Create an execution plan for:

Company: {workflow_input.company_description}
Product: {workflow_input.product_details}
Goals: {workflow_input.goals}
Constraints: {workflow_input.constraints}

Strategy:
GTM: {strategy.gtm_strategy[:400]}
Pricing: {strategy.pricing_strategy[:200]}
Positioning: {strategy.market_positioning[:200]}
Target Segment: {strategy.target_segment}

Recommendations:
{recs}

Growth Experiments:
{exps}

{context}

{_SCHEMA_INSTRUCTION}
"""
            raw_output = call_model(role=self.ROLE, prompt=prompt, system_prompt=_SYSTEM_PROMPT, temperature=0.2)
            result = self._parse_output(raw_output)

            self._memory.store(
                task_id=task_id, agent_name=self.AGENT_NAME,
                content=f"Timeline: {result.timeline}\nKPIs: {'; '.join(result.kpis[:4])}\nTasks: {len(result.tasks)}",
            )

            end_time = time.time()
            if workflow_state:
                workflow_state.mark_agent_done(self.AGENT_NAME, AgentStatus.SUCCESS)
                workflow_state.add_tokens(estimate_tokens(prompt), estimate_tokens(raw_output))

            log_execution(
                agent_name=self.AGENT_NAME, action="build_plan",
                start_time=start_time, end_time=end_time, status="success",
                metadata={"tasks": len(result.tasks), "sprints": len(result.sprints)},
            )
            logger.info(f"[{self.AGENT_NAME}] Plan created: {len(result.tasks)} tasks, {len(result.sprints)} sprints.")
            return result

        except Exception as exc:
            end_time = time.time()
            logger.error(f"[{self.AGENT_NAME}] Failed: {exc}")
            if workflow_state:
                workflow_state.mark_agent_done(self.AGENT_NAME, AgentStatus.FAILED, error=str(exc))
            log_execution(agent_name=self.AGENT_NAME, action="build_plan",
                          start_time=start_time, end_time=end_time, status="failed")
            return self._mock_output()

    def _parse_output(self, raw: str) -> ExecutionPlan:
        try:
            cleaned = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data = json.loads(cleaned)
            tasks = [Task(name=t.get("name","Task"), description=t.get("description",""),
                          owner=t.get("owner","Team"), duration=t.get("duration",""),
                          priority=t.get("priority","medium"), dependencies=t.get("dependencies",[]))
                     for t in data.get("tasks", []) if isinstance(t, dict)]
            sprints = [Sprint(sprint_number=s.get("sprint_number", i+1), name=s.get("name",f"Sprint {i+1}"),
                              duration=s.get("duration","2 weeks"), tasks=s.get("tasks",[]), goal=s.get("goal",""))
                       for i, s in enumerate(data.get("sprints", [])) if isinstance(s, dict)]
            milestones = [Milestone(name=m.get("name","Milestone"), target_date=m.get("target_date",""),
                                    success_criteria=m.get("success_criteria",""))
                          for m in data.get("milestones", []) if isinstance(m, dict)]
            return ExecutionPlan(tasks=tasks, timeline=data.get("timeline",""),
                                 kpis=data.get("kpis",[]), sprints=sprints, milestones=milestones,
                                 implementation_steps=data.get("implementation_steps",[]), is_mock=False)
        except Exception as exc:
            logger.warning(f"[{self.AGENT_NAME}] Parse failed: {exc}. Using mock.")
            return self._mock_output()

    def _mock_output(self) -> ExecutionPlan:
        return ExecutionPlan(
            tasks=[
                Task(name="Set up analytics", description="Integrate product analytics", owner="Engineering", duration="1 week", priority="high"),
                Task(name="Build onboarding", description="First-run experience", owner="Product", duration="2 weeks", priority="high"),
                Task(name="Launch referral program", description="Double-sided referral", owner="Growth", duration="1 week", priority="medium"),
            ],
            timeline="12 weeks from kickoff to public launch",
            kpis=["MAU: 500 by week 8", "Activation rate: >40%", "MRR: $10k by week 12", "NPS: >40"],
            sprints=[
                Sprint(sprint_number=1, name="Foundation", duration="2 weeks", tasks=["Set up analytics","Build onboarding"], goal="Beta ready"),
                Sprint(sprint_number=2, name="Growth Loops", duration="2 weeks", tasks=["Launch referral program"], goal="Growth channels active"),
            ],
            milestones=[
                Milestone(name="Beta Launch", target_date="Week 4", success_criteria="100 beta users"),
                Milestone(name="Public Launch", target_date="Week 8", success_criteria="500 MAU"),
            ],
            implementation_steps=["Week 1-2: Core product hardening", "Week 3-4: Beta onboarding", "Week 5-8: Growth launch"],
            is_mock=True,
        )


def run_planner(
    workflow_input: WorkflowInput,
    strategy: StrategyOutput,
    workflow_state: Optional[WorkflowState] = None,
) -> ExecutionPlan:
    return PlannerAgent().run(workflow_input, strategy, workflow_state)
