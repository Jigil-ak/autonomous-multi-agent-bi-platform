"""
Streamlit Dashboard
====================
Provides a live, demo-friendly UI for the Multi-Agent BI Platform.
Polls the FastAPI backend to render progressive updates and final reports.
"""

import time
import requests
import streamlit as st
from typing import Any

# Configuration
API_BASE_URL = "http://localhost:8000"
POLL_INTERVAL = 2.0

st.set_page_config(
    page_title="Multi-Agent BI Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "task_id" not in st.session_state:
    st.session_state.task_id = None
if "workflow_status" not in st.session_state:
    st.session_state.workflow_status = "IDLE"
if "final_report" not in st.session_state:
    st.session_state.final_report = None
if "logs" not in st.session_state:
    st.session_state.logs = []
if "metrics" not in st.session_state:
    st.session_state.metrics = {}

# ---------------------------------------------------------------------------
# API Helpers
# ---------------------------------------------------------------------------
def start_workflow(payload: dict) -> str:
    """Submit POST /analyze and return task_id."""
    try:
        resp = requests.post(f"{API_BASE_URL}/analyze", json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()["task_id"]
    except Exception as e:
        st.error(f"Failed to start workflow: {e}")
        return ""

def fetch_status(task_id: str) -> dict:
    """Poll GET /status/{task_id}."""
    try:
        resp = requests.get(f"{API_BASE_URL}/status/{task_id}", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e), "status": "ERROR"}

def fetch_logs(task_id: str) -> list:
    """Poll GET /logs/{task_id}."""
    try:
        resp = requests.get(f"{API_BASE_URL}/logs/{task_id}", timeout=5)
        if resp.status_code == 200:
            return resp.json().get("logs", [])
    except Exception:
        pass
    return []

def fetch_report(task_id: str) -> dict:
    """Poll GET /report/{task_id}."""
    try:
        resp = requests.get(f"{API_BASE_URL}/report/{task_id}", timeout=5)
        if resp.status_code == 200:
            return resp.json().get("report", {})
    except Exception:
        pass
    return {}

# ---------------------------------------------------------------------------
# UI Renderers
# ---------------------------------------------------------------------------
def render_sidebar():
    st.sidebar.title("🤖 Multi-Agent BI")
    st.sidebar.markdown("An autonomous multi-agent platform for strategic business intelligence.")
    
    if st.session_state.task_id:
        st.sidebar.subheader("Observability")
        st.sidebar.text(f"Task ID: {st.session_state.task_id}")
        st.sidebar.text(f"Status: {st.session_state.workflow_status}")
        
        m = st.session_state.metrics
        if m:
            st.sidebar.metric("Agents Completed", len(m.get("completed_agents", [])))
            st.sidebar.metric("Est. Tokens", m.get("token_estimate", 0))
            if m.get("fallback_events"):
                st.sidebar.warning("Fallback Activated (Pro ➔ Flash)")
                for ev in m.get("fallback_events", []):
                    st.sidebar.caption(f"⚠ {ev}")

def render_input_section():
    st.title("Autonomous Multi-Agent Business Intelligence")
    
    with st.form("workflow_input"):
        st.markdown("### Strategic Request Context")
        company = st.text_input("Company Name / Description", "An AI-powered SaaS analytics platform")
        product = st.text_area("Product Details", "Self-service BI dashboard with natural language queries")
        audience = st.text_input("Target Audience", "Small and medium businesses")
        goals = st.text_input("Goals", "Acquire 500 paying users within 6 months")
        constraints = st.text_input("Constraints", "Bootstrap budget, team of 5")
        
        submitted = st.form_submit_button("Generate Strategy", type="primary")
        
        if submitted:
            st.session_state.task_id = None
            st.session_state.workflow_status = "STARTING"
            st.session_state.final_report = None
            st.session_state.logs = []
            
            payload = {
                "company_description": company,
                "product_details": product,
                "target_audience": audience,
                "goals": goals,
                "constraints": constraints,
                "enable_human_approval": False
            }
            task_id = start_workflow(payload)
            if task_id:
                st.session_state.task_id = task_id
                st.session_state.workflow_status = "RUNNING"
                st.rerun()

def render_active_workflow():
    task_id = st.session_state.task_id
    st.subheader(f"Workflow Running: {task_id}")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    active_agent_display = st.empty()
    
    logs_expander = st.expander("System Logs", expanded=True)
    logs_container = logs_expander.empty()
    
    while st.session_state.workflow_status == "RUNNING":
        status_data = fetch_status(task_id)
        current_status = status_data.get("status", "ERROR")
        
        if current_status == "ERROR":
            st.error("Lost connection to API or task failed.")
            st.session_state.workflow_status = "FAILED"
            break
            
        progress = status_data.get("progress_percent", 0)
        progress_bar.progress(progress / 100.0)
        
        stage = status_data.get("stage", "unknown").upper()
        status_text.markdown(f"**Stage:** {stage}")
        
        active_agent = status_data.get("active_agent")
        if active_agent:
            active_agent_display.info(f"🟡 {active_agent.replace('_', ' ').title()} is working...")
        else:
            active_agent_display.empty()
            
        logs = fetch_logs(task_id)
        if logs:
            st.session_state.logs = logs
            logs_str = "\n".join([f"[{l['ts']}] {l['msg']}" for l in logs])
            logs_container.code(logs_str, language="log")
            
        st.session_state.metrics = status_data
        
        if current_status in ["COMPLETED", "COMPLETED_WITH_WARNINGS", "FAILED"]:
            st.session_state.workflow_status = current_status
            break
            
        time.sleep(POLL_INTERVAL)
        st.rerun()

def render_report(report: dict):
    st.success("Workflow Completed Successfully!")
    
    research = report.get("research", {})
    strategy = report.get("strategy", {})
    plan = report.get("execution_plan", {})
    critique = report.get("critique", {})
    qa = report.get("qa_report", {})
    
    st.header("Executive Report")
    
    with st.expander("📊 Market Research & Competitors", expanded=True):
        st.markdown(f"**Summary:** {research.get('research_summary', 'N/A')}")
        st.markdown("#### Top Competitors")
        for c in research.get("competitors", []):
            st.markdown(f"- **{c.get('name')}**: {c.get('description')} *(Pricing: {c.get('pricing')})*")
        
        cols = st.columns(2)
        with cols[0]:
            st.markdown("#### Market Signals")
            for sig in research.get("market_signals", []):
                st.markdown(f"- {sig}")
        with cols[1]:
            st.markdown("#### Trends")
            for t in research.get("trends", []):
                st.markdown(f"- {t}")

    with st.expander("🎯 Go-To-Market Strategy", expanded=True):
        st.markdown("#### GTM Strategy")
        st.write(strategy.get("gtm_strategy", "N/A"))
        st.markdown("#### Pricing")
        st.write(strategy.get("pricing_strategy", "N/A"))
        st.markdown("#### Positioning")
        st.write(strategy.get("market_positioning", "N/A"))
        
        st.markdown("#### Top Recommendations")
        for r in strategy.get("recommendations", []):
            st.markdown(f"- {r}")
            
    with st.expander("📈 Growth Experiments", expanded=False):
        for exp in strategy.get("growth_experiments", []):
            st.markdown(f"**{exp.get('name')}** (Priority: {exp.get('priority')})")
            st.write(exp.get('description'))
            st.caption(f"Expected Outcome: {exp.get('expected_outcome')}")
            st.divider()

    with st.expander("🗓️ Execution Plan", expanded=False):
        st.markdown(f"**Timeline:** {plan.get('timeline', 'N/A')}")
        st.markdown("#### KPIs")
        for k in plan.get("kpis", []):
            st.markdown(f"- {k}")
            
        st.markdown("#### Sprints")
        for s in plan.get("sprints", []):
            st.markdown(f"**Sprint {s.get('sprint_number')}: {s.get('name')}** ({s.get('duration')})")
            st.write(f"*Goal: {s.get('goal')}*")
            st.write(f"Tasks: {', '.join(s.get('tasks', []))}")
            st.divider()

    with st.expander("🔍 QA & Critic Validation", expanded=False):
        st.markdown(f"**QA Verdict:** {qa.get('overall_verdict', 'N/A')} (Score: {qa.get('completeness_score', 0)})")
        st.markdown(f"**Critic Quality Score:** {critique.get('overall_quality_score', 0)}")
        st.markdown("#### Critic Issues Identified")
        for iss in critique.get("issues", []):
            st.markdown(f"- **{iss.get('section')}** ({iss.get('severity')}): {iss.get('description')}")
            st.caption(f"Suggestion: {iss.get('suggestion')}")

    st.markdown("---")
    st.subheader("System Performance")
    ws = report.get("workflow_state", {})
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Agents Completed", len(ws.get("completed_agents", [])))
    col2.metric("Tokens Used", ws.get("total_tokens", 0))
    col3.metric("QA Verdict", qa.get('overall_verdict', 'N/A'))
    col4.metric("Quality Score", critique.get('overall_quality_score', 0))

# ---------------------------------------------------------------------------
# Main App Flow
# ---------------------------------------------------------------------------
def main():
    render_sidebar()
    
    if st.session_state.workflow_status == "IDLE":
        render_input_section()
        
    elif st.session_state.workflow_status == "RUNNING":
        render_active_workflow()
        
    elif st.session_state.workflow_status in ["COMPLETED", "COMPLETED_WITH_WARNINGS", "FAILED"]:
        if st.session_state.workflow_status == "FAILED":
            st.error("Workflow failed to complete.")
            if st.button("Start Over"):
                st.session_state.workflow_status = "IDLE"
                st.rerun()
        else:
            if not st.session_state.final_report:
                with st.spinner("Fetching final report..."):
                    st.session_state.final_report = fetch_report(st.session_state.task_id)
            
            if st.session_state.workflow_status == "COMPLETED_WITH_WARNINGS":
                st.warning("Workflow completed with warnings (fallback models may have been used).")
                
            if st.session_state.final_report:
                render_report(st.session_state.final_report)
            
            if st.button("Start New Analysis"):
                st.session_state.task_id = None
                st.session_state.workflow_status = "IDLE"
                st.session_state.final_report = None
                st.rerun()
                
            logs_expander = st.expander("View Execution Logs", expanded=False)
            logs_str = "\n".join([f"[{l['ts']}] {l['msg']}" for l in st.session_state.logs])
            logs_expander.code(logs_str, language="log")

if __name__ == "__main__":
    main()
