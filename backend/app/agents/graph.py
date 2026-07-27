import uuid
import logging
from typing import Dict, Any
from app.agents.state import AgentState
from app.agents.stage1_intake.intake_parser import IntakeParsingAgent
from app.agents.stage2_analysis.match_ats import MatchATSAgent
from app.agents.stage2_analysis.code_portfolio import CodePortfolioAgent
from app.agents.stage2_analysis.skill_insight import SkillInsightAgent
from app.agents.stage3_strategy.optimizer import OptimizerAgent
from app.agents.stage3_strategy.career_trajectory import CareerTrajectoryAgent
from app.agents.stage3_strategy.interview_coach import InterviewCoachAgent
from app.agents.stage4_quality.quality_gate import QualityGateAgent
from app.agents.stage4_quality.report_generator import ReportGeneratorAgent
from app.core.monitoring import MonitoringService

logger = logging.getLogger("agent_graph")

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger.info("LangGraph package not installed globally. Running sequential state machine fallback.")

# Node Functions
def run_intake_node(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    state = AgentState(**state_dict)
    updated_state = IntakeParsingAgent.parse(state)
    MonitoringService.log_agent_call(state.report_id, "Intake & Parsing Agent", tier=0, model_name="gemini-2.5-flash")
    return updated_state.model_dump()

def run_match_ats_node(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    state = AgentState(**state_dict)
    updated_state = MatchATSAgent.analyze(state)
    MonitoringService.log_agent_call(state.report_id, "Match & ATS Agent", tier=1, model_name="deepseek-chat")
    return updated_state.model_dump()

def run_code_portfolio_node(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    state = AgentState(**state_dict)
    updated_state = CodePortfolioAgent.analyze(state)
    MonitoringService.log_agent_call(state.report_id, "Code & Portfolio Agent", tier=1, model_name="deepseek-chat")
    return updated_state.model_dump()

def run_skill_insight_node(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    state = AgentState(**state_dict)
    updated_state = SkillInsightAgent.analyze(state)
    return updated_state.model_dump()

def run_optimizer_node(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    state = AgentState(**state_dict)
    updated_state = OptimizerAgent.generate_variants(state)
    MonitoringService.log_agent_call(state.report_id, "Optimizer Agent", tier=2, model_name="claude-3.5-sonnet")
    return updated_state.model_dump()

def run_career_trajectory_node(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    state = AgentState(**state_dict)
    updated_state = CareerTrajectoryAgent.analyze(state)
    return updated_state.model_dump()

def run_interview_coach_node(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    state = AgentState(**state_dict)
    updated_state = InterviewCoachAgent.prepare(state)
    return updated_state.model_dump()

def run_quality_gate_node(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    state = AgentState(**state_dict)
    updated_state = QualityGateAgent.evaluate(state)
    return updated_state.model_dump()

def run_report_generator_node(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    state = AgentState(**state_dict)
    updated_state = ReportGeneratorAgent.generate(state)
    MonitoringService.log_agent_call(state.report_id, "Report Generator Agent", tier=2, model_name="claude-3.5-sonnet")
    return updated_state.model_dump()

def execute_career_pipeline(
    raw_resume_text: str,
    raw_jd_text: str,
    github_url: str = None,
    portfolio_url: str = None
) -> Dict[str, Any]:
    """Execute complete 10-Agent workflow."""
    report_id = f"rpt_{uuid.uuid4().hex[:8]}"
    
    initial_state = AgentState(
        report_id=report_id,
        raw_resume_text=raw_resume_text,
        raw_jd_text=raw_jd_text,
        github_url=github_url,
        portfolio_url=portfolio_url
    )
    
    if LANGGRAPH_AVAILABLE:
        try:
            workflow = StateGraph(dict)
            workflow.add_node("intake_parser", run_intake_node)
            workflow.add_node("match_ats", run_match_ats_node)
            workflow.add_node("code_portfolio", run_code_portfolio_node)
            workflow.add_node("skill_insight", run_skill_insight_node)
            workflow.add_node("optimizer", run_optimizer_node)
            workflow.add_node("career_trajectory", run_career_trajectory_node)
            workflow.add_node("interview_coach", run_interview_coach_node)
            workflow.add_node("quality_gate", run_quality_gate_node)
            workflow.add_node("report_generator", run_report_generator_node)

            workflow.set_entry_point("intake_parser")
            workflow.add_edge("intake_parser", "match_ats")
            workflow.add_edge("match_ats", "code_portfolio")
            workflow.add_edge("code_portfolio", "skill_insight")
            workflow.add_edge("skill_insight", "optimizer")
            workflow.add_edge("optimizer", "career_trajectory")
            workflow.add_edge("career_trajectory", "interview_coach")
            workflow.add_edge("interview_coach", "quality_gate")
            workflow.add_edge("quality_gate", "report_generator")
            workflow.add_edge("report_generator", END)

            compiled_graph = workflow.compile()
            final_output_state = compiled_graph.invoke(initial_state.model_dump())
            return final_output_state.get("final_report", {})
        except Exception as e:
            logger.warning(f"LangGraph execution fallback triggered ({e}).")

    # Direct sequential agent state machine execution
    s = run_intake_node(initial_state.model_dump())
    s = run_match_ats_node(s)
    s = run_code_portfolio_node(s)
    s = run_skill_insight_node(s)
    s = run_optimizer_node(s)
    s = run_career_trajectory_node(s)
    s = run_interview_coach_node(s)
    s = run_quality_gate_node(s)
    s = run_report_generator_node(s)
    
    return s.get("final_report", {})
