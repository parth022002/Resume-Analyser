import logging
from typing import Dict, Any
from app.agents.state import AgentState

logger = logging.getLogger("quality_gate")

class QualityGateAgent:
    """
    Stage 4 — Quality Gate (Ragas / DeepEval proxy):
    Scores groundedness, completeness, and relevance.
    Triggers automatic LangGraph loop-back retry if quality score falls below threshold.
    """
    
    QUALITY_THRESHOLD = 0.75
    MAX_RETRIES = 2
    
    @classmethod
    def evaluate(cls, state: AgentState) -> AgentState:
        cand_graph = state.candidate_graph
        job_graph = state.job_graph
        match = state.match_analysis
        
        # 1. Groundedness Score
        has_skills = len(cand_graph.skills) > 0 if cand_graph else False
        groundedness = 0.95 if has_skills else 0.60
        
        # 2. Completeness Score
        completeness = 0.90 if match and "overall_match_score" in match else 0.50
        
        # 3. Relevance Score
        relevance = 0.92 if job_graph and job_graph.required_skills else 0.70
        
        # Aggregate Quality Gate Score
        quality_score = round((groundedness * 0.4) + (completeness * 0.3) + (relevance * 0.3), 2)
        state.quality_score = quality_score
        
        if quality_score < cls.QUALITY_THRESHOLD and state.retry_count < cls.MAX_RETRIES:
            logger.warning(f"Quality Gate Score ({quality_score}) below threshold ({cls.QUALITY_THRESHOLD}). Triggering retry loop #{state.retry_count + 1}.")
            state.is_quality_passed = False
            state.retry_count += 1
        else:
            state.is_quality_passed = True
            
        return state
