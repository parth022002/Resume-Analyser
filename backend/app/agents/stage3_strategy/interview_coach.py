from typing import Dict, Any, List
from app.agents.state import AgentState

class InterviewCoachAgent:
    """
    Stage 3 — Interview Coach Agent:
    Generates technical, behavioral, and company-specific interview prep questions,
    coding challenge prompts, and computes candidate Interview Readiness Score.
    """
    
    @classmethod
    def prepare(cls, state: AgentState) -> AgentState:
        cand_graph = state.candidate_graph
        job_graph = state.job_graph
        match = state.match_analysis
        
        target_role = job_graph.title if job_graph else "Senior Software Engineer"
        company = job_graph.company if job_graph else "Target Tech Company"
        match_score = match.get("overall_match_score", 75.0)
        
        # Calculate Interview Readiness Score
        readiness_score = round(min(max((match_score * 0.85) + 10.0, 60.0), 96.0), 1)

        interview_prep_data = {
            "readiness_score": readiness_score,
            "readiness_label": "High Readiness" if readiness_score >= 80 else "Moderate Prep Required",
            "technical_questions": [
                {
                    "question": f"How do you design and optimize high-throughput REST / Async APIs in a {target_role} architecture?",
                    "focus": "API Performance & Concurrency",
                    "sample_answer_hint": "Discuss connection pooling, caching strategies (Redis), asynchronous task queues (Celery), and database index tuning."
                },
                {
                    "question": "Walk me through how you handle database migrations and zero-downtime schema updates in production.",
                    "focus": "Database Engineering",
                    "sample_answer_hint": "Explain additive schema changes, blue-green deployments, and backward-compatible database migrations."
                }
            ],
            "behavioral_questions": [
                {
                    "question": "Describe a situation where a critical production bug occurred right before a release. How did you diagnose and resolve it under pressure?",
                    "framework": "STAR Method (Situation, Task, Action, Result)",
                    "tip": "Focus 60% of your answer on your specific troubleshooting steps, log analysis, and root cause prevention."
                },
                {
                    "question": "Tell me about a time you had a technical disagreement with a team member regarding system architecture.",
                    "framework": "STAR Method",
                    "tip": "Highlight data-driven decision making, benchmarking, and collaborative resolution."
                }
            ],
            "company_specific_prep": {
                "target_company": company,
                "focus_areas": [
                    f"Understand {company}'s core product architecture and engineering culture.",
                    "Review recent engineering blog posts and public tech talks given by the company.",
                    "Prepare 3 strategic questions to ask the hiring manager about team growth and technical roadmap."
                ]
            },
            "coding_challenges": [
                {
                    "title": "Design a Distributed Rate Limiter",
                    "difficulty": "Medium",
                    "concept": "Sliding Window Counter / Token Bucket algorithm using Redis."
                },
                {
                    "title": "LRU Cache Implementation",
                    "difficulty": "Medium",
                    "concept": "Doubly Linked List + Hash Map O(1) time complexity."
                }
            ]
        }
        
        state.interview_prep = interview_prep_data
        return state
