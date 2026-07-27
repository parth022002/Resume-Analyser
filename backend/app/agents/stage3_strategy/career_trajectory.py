from typing import Dict, Any, List
from app.agents.state import AgentState

class CareerTrajectoryAgent:
    """
    Stage 3 — Career Trajectory Agent:
    Generates industry trend insights, career gap analysis, and a 30/90/180-day learning roadmap.
    """
    
    @classmethod
    def analyze(cls, state: AgentState) -> AgentState:
        cand_graph = state.candidate_graph
        job_graph = state.job_graph
        match = state.match_analysis
        missing = match.get("missing_skills", [])
        
        target_role = job_graph.title if job_graph else "Senior Software Engineer"
        
        # 30-Day Plan (Immediate Gaps & Core Fundamentals)
        day_30_goals = [
            f"Master core concepts for high-priority missing skill: {missing[0] if missing else 'System Design'}.",
            "Build 1 hands-on prototype project demonstrating API integration & containerization.",
            "Refactor top GitHub repository with structured README, tests, and documentation."
        ]
        
        # 90-Day Plan (Advanced Integration & Architecture)
        day_90_goals = [
            f"Gain proficiency in cloud deployment & orchestration ({missing[1] if len(missing) > 1 else 'Kubernetes'}).",
            "Contribute to open-source software or build a full-stack portfolio showcase.",
            "Conduct mock technical interview sessions focusing on system architecture & data structures."
        ]

        # 180-Day Plan (Seniority & Career Scaling)
        day_180_goals = [
            f"Achieve interview readiness for senior-level {target_role} roles.",
            "Publish a technical blog post or system design case study highlighting production engineering challenges solved.",
            "Target top tier enterprise and high-growth technology companies for placement."
        ]

        career_trajectory_data = {
            "target_role": target_role,
            "industry_trends": [
                f"High market demand for full-stack engineers with {target_role} capabilities.",
                "Increased focus on AI-assisted development tools, API performance, and cloud-native architecture.",
                "Strong preference for candidates who showcase public code proof and active GitHub repos."
            ],
            "roadmap": {
                "day_30": {
                    "phase": "Phase 1: Fundamental Gap Closure (Days 1–30)",
                    "focus": "Core Technical Proficiency",
                    "milestones": day_30_goals
                },
                "day_90": {
                    "phase": "Phase 2: Advanced Architecture & Portfolio (Days 31–90)",
                    "focus": "Production System Engineering",
                    "milestones": day_90_goals
                },
                "day_180": {
                    "phase": "Phase 3: Seniority & Market Placement (Days 91–180)",
                    "focus": "Interview Mastery & Role Transition",
                    "milestones": day_180_goals
                }
            }
        }
        
        state.career_trajectory = career_trajectory_data
        return state
