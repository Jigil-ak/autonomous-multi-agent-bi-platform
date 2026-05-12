"""
Centralized Model Router
=========================
Single entry point for all Gemini API calls across the platform.

Features:
- Role-based model selection (strategy → pro, research → flash, etc.)
- Automatic fallback chain when primary model fails or is rate-limited
- Mock response support for demo resilience
- Integration with retry handler for exponential backoff
- Token usage estimation logging

IMPORTANT: No other file should instantiate a Gemini model directly.
           Always use ``get_model()`` or ``call_model()`` from this module.
"""

import time
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from config import (
    GOOGLE_API_KEY,
    MODEL_ROUTER,
    MODEL_FALLBACKS,
    RETRY_DELAYS,
    MAX_RETRIES,
)
from utils.logger import get_logger, log_execution, estimate_tokens, ExecutionTimer

logger = get_logger("model_router")


# ---------------------------------------------------------------------------
# Mock Responses (used when all API calls fail during demos)
# ---------------------------------------------------------------------------
MOCK_RESPONSES: dict[str, str] = {
    "research": (
        '{"competitors": ["Competitor A", "Competitor B"], '
        '"market_signals": ["Growing market demand", "Shift to AI-first solutions"], '
        '"trends": ["Automation", "Self-service analytics"], '
        '"audience_insights": "Target audience prefers ease of use over feature depth.", '
        '"pricing_data": "Market average is $49-199/month per seat."}'
    ),
    "strategy": (
        '{"gtm_strategy": "Launch with a product-led growth model targeting SMBs.", '
        '"pricing_strategy": "Freemium tier with usage-based premium plans.", '
        '"growth_experiments": ["Referral program", "Content-led SEO", "Partnership integrations"], '
        '"positioning": "The simplest AI-powered BI tool for growing teams.", '
        '"recommendations": ["Focus on onboarding experience", "Build integrations marketplace"]}'
    ),
    "planning": (
        '{"tasks": [{"name": "MVP Launch", "duration": "4 weeks"}, '
        '{"name": "Beta Program", "duration": "6 weeks"}], '
        '"timeline": "Q1-Q2 2025", '
        '"kpis": ["Monthly Active Users", "Activation Rate", "NPS Score"], '
        '"sprints": ["Sprint 1: Core features", "Sprint 2: Integrations", "Sprint 3: Polish"], '
        '"milestones": ["Alpha release", "100 beta users", "Public launch"]}'
    ),
    "critic": (
        '{"issues": ["Strategy lacks specifics on customer acquisition cost targets"], '
        '"suggestions": ["Add CAC/LTV analysis", "Include risk mitigation plan"], '
        '"severity_scores": {"strategy_gaps": 3, "data_quality": 2}, '
        '"improved_sections": "Consider adding competitive moat analysis."}'
    ),
    "qa": (
        '{"completeness_score": 0.85, '
        '"formatting_issues": [], '
        '"section_scores": {"research": 0.9, "strategy": 0.8, "planning": 0.85}, '
        '"overall_verdict": "PASS — Report meets quality standards with minor gaps."}'
    ),
    "memory": "Context stored and summarized successfully.",
    "orchestrator": "Workflow step completed successfully.",
}


# ---------------------------------------------------------------------------
# Model Factory
# ---------------------------------------------------------------------------
def get_model(
    role: str,
    temperature: float = 0.3,
    model_name_override: str | None = None,
) -> ChatGoogleGenerativeAI:
    """
    Create a ``ChatGoogleGenerativeAI`` instance for the given agent role.

    Parameters
    ----------
    role : str
        Agent role key (e.g. ``'strategy'``, ``'research'``).
    temperature : float
        Sampling temperature for the model.
    model_name_override : str, optional
        Override the model name from the routing table.

    Returns
    -------
    ChatGoogleGenerativeAI
        A configured LangChain chat model.
    """
    model_name = model_name_override or MODEL_ROUTER.get(role, "gemini-2.5-flash")
    logger.info(f"Initializing model '{model_name}' for role '{role}'")

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=GOOGLE_API_KEY,
        temperature=temperature,
        convert_system_message_to_human=True,
    )


# ---------------------------------------------------------------------------
# Unified Model Call
# ---------------------------------------------------------------------------
def call_model(
    role: str,
    prompt: str,
    system_prompt: str | None = None,
    temperature: float = 0.3,
    use_fallback: bool = True,
    use_mock: bool = True,
) -> str:
    """
    Single entry point for all LLM calls across the platform.

    Attempts the primary model for the role, then walks the fallback chain,
    and finally returns a mock response if everything fails.

    Parameters
    ----------
    role : str
        Agent role key.
    prompt : str
        The user / task prompt.
    system_prompt : str, optional
        System-level instruction prepended to the conversation.
    temperature : float
        Sampling temperature.
    use_fallback : bool
        Whether to try fallback models on failure.
    use_mock : bool
        Whether to return a mock response if all models fail.

    Returns
    -------
    str
        The model's response text.
    """
    primary_model_name = MODEL_ROUTER.get(role, "gemini-2.5-flash")
    models_to_try = [primary_model_name]

    if use_fallback:
        models_to_try.extend(MODEL_FALLBACKS.get(primary_model_name, []))

    # Build messages
    messages = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=prompt))

    # Estimate input tokens for logging
    input_tokens = estimate_tokens(prompt + (system_prompt or ""))

    for model_name in models_to_try:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                with ExecutionTimer() as timer:
                    model = ChatGoogleGenerativeAI(
                        model=model_name,
                        google_api_key=GOOGLE_API_KEY,
                        temperature=temperature,
                        convert_system_message_to_human=True,
                    )
                    response = model.invoke(messages)

                response_text = response.content
                output_tokens = estimate_tokens(response_text)

                log_execution(
                    agent_name=f"model_router.{role}",
                    action="call_model",
                    start_time=timer.start,
                    end_time=timer.end,
                    status="success" if model_name == primary_model_name else "fallback",
                    metadata={
                        "model": model_name,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "attempt": attempt,
                    },
                )
                return response_text

            except Exception as exc:
                delay = RETRY_DELAYS[(attempt - 1) % len(RETRY_DELAYS)]
                logger.warning(
                    f"Model '{model_name}' attempt {attempt}/{MAX_RETRIES} "
                    f"for role '{role}' failed: {exc}. "
                    f"Retrying in {delay}s..."
                )
                time.sleep(delay)

        # Log that this model is exhausted, move to next fallback
        logger.warning(
            f"All retries exhausted for model '{model_name}' (role: {role}). "
            f"Trying next fallback..."
        )

    # All models failed — return mock response if enabled
    if use_mock and role in MOCK_RESPONSES:
        logger.warning(
            f"All models failed for role '{role}'. "
            f"Returning mock response for demo resilience."
        )
        return MOCK_RESPONSES[role]

    # Nothing worked
    error_msg = (
        f"All models and fallbacks exhausted for role '{role}'. "
        f"No mock response available."
    )
    logger.error(error_msg)
    raise RuntimeError(error_msg)


# ---------------------------------------------------------------------------
# Async Model Call
# ---------------------------------------------------------------------------
async def acall_model(
    role: str,
    prompt: str,
    system_prompt: str | None = None,
    temperature: float = 0.3,
    use_fallback: bool = True,
    use_mock: bool = True,
) -> str:
    """
    Async version of :func:`call_model`.

    Uses ``ainvoke`` for non-blocking execution in FastAPI endpoints.
    """
    import asyncio

    primary_model_name = MODEL_ROUTER.get(role, "gemini-2.5-flash")
    models_to_try = [primary_model_name]

    if use_fallback:
        models_to_try.extend(MODEL_FALLBACKS.get(primary_model_name, []))

    messages = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=prompt))

    input_tokens = estimate_tokens(prompt + (system_prompt or ""))

    for model_name in models_to_try:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                with ExecutionTimer() as timer:
                    model = ChatGoogleGenerativeAI(
                        model=model_name,
                        google_api_key=GOOGLE_API_KEY,
                        temperature=temperature,
                        convert_system_message_to_human=True,
                    )
                    response = await model.ainvoke(messages)

                response_text = response.content
                output_tokens = estimate_tokens(response_text)

                log_execution(
                    agent_name=f"model_router.{role}",
                    action="acall_model",
                    start_time=timer.start,
                    end_time=timer.end,
                    status="success" if model_name == primary_model_name else "fallback",
                    metadata={
                        "model": model_name,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "attempt": attempt,
                    },
                )
                return response_text

            except Exception as exc:
                delay = RETRY_DELAYS[(attempt - 1) % len(RETRY_DELAYS)]
                logger.warning(
                    f"[async] Model '{model_name}' attempt {attempt}/{MAX_RETRIES} "
                    f"for role '{role}' failed: {exc}. "
                    f"Retrying in {delay}s..."
                )
                await asyncio.sleep(delay)

        logger.warning(
            f"[async] All retries exhausted for model '{model_name}' "
            f"(role: {role}). Trying next fallback..."
        )

    if use_mock and role in MOCK_RESPONSES:
        logger.warning(
            f"[async] All models failed for role '{role}'. "
            f"Returning mock response for demo resilience."
        )
        return MOCK_RESPONSES[role]

    error_msg = (
        f"All models and fallbacks exhausted for role '{role}'. "
        f"No mock response available."
    )
    logger.error(error_msg)
    raise RuntimeError(error_msg)
