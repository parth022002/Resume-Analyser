from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class CandidateKnowledgeGraph(BaseModel):
    """
    Unified Candidate Knowledge Graph built in Stage 1.
    All downstream agents consume this structured graph.
    """
    name: str = "Candidate"
    email: str = "N/A"
    phone: str = "N/A"
    location: str = "N/A"
    summary: str = ""
    skills: List[str] = Field(default_factory=list)
    experience: List[Dict[str, Any]] = Field(default_factory=list)
    education: List[Dict[str, Any]] = Field(default_factory=list)
    projects: List[Dict[str, Any]] = Field(default_factory=list)
    github_url: Optional[str] = None
    portfolio_url: Optional[str] = None
    total_experience_years: float = 0.0

class JobKnowledgeGraph(BaseModel):
    """Structured representation of target Job Description."""
    title: str = "Target Position"
    company: str = "Target Company"
    required_skills: List[str] = Field(default_factory=list)
    preferred_skills: List[str] = Field(default_factory=list)
    min_experience_years: float = 0.0
    responsibilities: List[str] = Field(default_factory=list)
    domain: str = "General Engineering"

class AgentState(BaseModel):
    """LangGraph Shared State across all 10 agents."""
    report_id: str
    raw_resume_text: str
    raw_jd_text: str
    github_url: Optional[str] = None
    portfolio_url: Optional[str] = None
    
    # Security Flag
    guardrails_passed: bool = True
    guardrail_warnings: List[str] = Field(default_factory=list)
    
    # Knowledge Graphs (Stage 1)
    candidate_graph: Optional[CandidateKnowledgeGraph] = None
    job_graph: Optional[JobKnowledgeGraph] = None
    
    # Analysis & Strategy Outputs (Stages 2 & 3)
    match_analysis: Dict[str, Any] = Field(default_factory=dict)
    skill_insights: Dict[str, Any] = Field(default_factory=dict)
    code_review: Dict[str, Any] = Field(default_factory=dict)
    resume_variants: Dict[str, Any] = Field(default_factory=dict)
    career_trajectory: Dict[str, Any] = Field(default_factory=dict)
    interview_prep: Dict[str, Any] = Field(default_factory=dict)
    
    # Quality & Explainability (Stage 4)
    explainability: List[Dict[str, Any]] = Field(default_factory=list)
    quality_score: float = 1.0
    is_quality_passed: bool = True
    retry_count: int = 0
    
    # Final Output
    final_report: Dict[str, Any] = Field(default_factory=dict)
