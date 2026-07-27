from typing import Dict, Any, List
from app.agents.state import AgentState

class SkillInsightAgent:
    """
    Stage 2 — Skill & Requirement Insight Agent:
    Finds missing and emerging skills and explains contextual importance:
    real-world application, interview expectations, and market trends.
    """
    
    @classmethod
    def analyze(cls, state: AgentState) -> AgentState:
        match = state.match_analysis
        missing_skills = match.get("missing_skills", [])
        
        insights = []
        for skill in missing_skills[:5]:
            insights.append(cls._explain_skill(skill))
            
        if not insights:
            insights.append({
                "skill": "System Architecture & Design",
                "importance": "High Priority",
                "real_world_application": "Designing scalable distributed systems and resilient microservice architectures.",
                "interview_expectation": "Expect senior system design questions regarding load balancing, caching, and data partitioning.",
                "market_trend": "Growing demand for cloud-native design across top tech firms."
            })

        state.skill_insights = {
            "missing_skill_count": len(missing_skills),
            "priority_insights": insights
        }
        return state

    @staticmethod
    def _explain_skill(skill: str) -> Dict[str, Any]:
        knowledge_base = {
            "docker": {
                "skill": "Docker & Containerization",
                "importance": "Critical Core Skill",
                "real_world_application": "Packaging microservices into isolated containers ensuring environment consistency between dev and production.",
                "interview_expectation": "Be prepared to explain Dockerfile layering, multi-stage builds, and container networking.",
                "market_trend": "Industry standard requirement for 92%+ of full-stack backend roles."
            },
            "kubernetes": {
                "skill": "Kubernetes Orchestration",
                "importance": "High Growth Skill",
                "real_world_application": "Automating container deployment, autoscaling, and cluster management.",
                "interview_expectation": "Understand pods, deployments, ingress controllers, and zero-downtime rolling updates.",
                "market_trend": "High demand for enterprise cloud-native infrastructure engineering."
            },
            "aws cloud": {
                "skill": "AWS Cloud Infrastructure",
                "importance": "Core Infrastructure Requirement",
                "real_world_application": "Deploying serverless endpoints, S3 storage, and managing IAM security policies.",
                "interview_expectation": "Explain EC2 vs ECS serverless, Lambda triggers, and database provisioning.",
                "market_trend": "Dominant public cloud provider across enterprise software platforms."
            },
            "system design": {
                "skill": "System Architecture & Design",
                "importance": "High Seniority Indicator",
                "real_world_application": "Architecting resilient, fault-tolerant software systems handling high concurrency.",
                "interview_expectation": "Demonstrate understanding of CAP theorem, database sharding, and caching strategies.",
                "market_trend": "Key differentiator for senior software engineering career progression."
            }
        }
        
        lower_skill = skill.lower()
        if lower_skill in knowledge_base:
            return knowledge_base[lower_skill]
            
        return {
            "skill": skill,
            "importance": "Recommended Requirement",
            "real_world_application": f"Utilizing {skill} to streamline development workflows and enhance feature capabilities.",
            "interview_expectation": f"Expect technical screening questions evaluating practical experience with {skill}.",
            "market_trend": f"Increasing adoption of {skill} in modern software engineering stacks."
        }
