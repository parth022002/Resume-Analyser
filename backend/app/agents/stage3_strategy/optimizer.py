from typing import Dict, Any, List
from app.agents.state import AgentState
from app.rag.retriever import HybridRetriever
from app.core.model_router import ModelRouter

class OptimizerAgent:
    """
    Stage 3 — Optimizer Agent:
    Generates ATS-Optimized, Technical Deep-Dive, and Executive Resume Variants.
    """
    
    @classmethod
    def generate_variants(cls, state: AgentState) -> AgentState:
        cand_graph = state.candidate_graph
        job_graph = state.job_graph
        match = state.match_analysis
        
        # Search RAG corpus for optimization guidelines
        rag_context = HybridRetriever.search(f"{job_graph.title} {', '.join(match.get('missing_skills', []))}", top_k=2)
        
        name = cand_graph.name if cand_graph else "Candidate"
        role = job_graph.title if job_graph else "Software Engineer"
        matching_skills = ", ".join(match.get("matching_skills", ["Software Engineering"]))
        missing_skills = ", ".join(match.get("missing_skills", ["Cloud Computing"]))
        
        # 1. ATS-Optimized Variant
        ats_variant = {
            "title": "ATS-Optimized Variant",
            "target": "Maximum Screener Keyword Density & Readability",
            "summary": f"Results-driven {role} specializing in {matching_skills}. Proven track record in delivering scalable applications and implementing clean engineering practices aligned with {job_graph.company} standards.",
            "highlighted_skills": match.get("matching_skills", []) + match.get("missing_skills", [])[:3],
            "key_bullet_points": [
                f"Developed high-availability backend microservices using {matching_skills}.",
                f"Integrated automated testing and CI/CD pipelines to ensure seamless deployment.",
                f"Collaborated with engineering teams to incorporate {missing_skills[:30]} best practices."
            ]
        }

        # 2. Technical Deep-Dive Variant
        technical_variant = {
            "title": "Technical Deep-Dive Variant",
            "target": "Engineering Hiring Managers & System Architects",
            "summary": f"Hands-on {role} with expertise in building robust, low-latency distributed systems using {matching_skills}. Focus on API performance, concurrency, and scalable database design.",
            "highlighted_skills": cand_graph.skills if cand_graph else ["Python", "FastAPI", "React"],
            "key_bullet_points": [
                f"Architected asynchronous service endpoints using {matching_skills}, optimizing execution speed.",
                "Engineered scalable data models and vector search indexing for high-throughput queries.",
                "Implemented rigorous code review standards and unit/integration testing coverage."
            ]
        }

        # 3. Executive / Leadership Variant
        executive_variant = {
            "title": "Executive & Leadership Variant",
            "target": "Directors, VP of Engineering & C-Suite Executives",
            "summary": f"Strategic {role} with a focus on technical roadmap execution, system stability, and cross-functional team leadership. Proven success translating business objectives into high-impact software solutions.",
            "highlighted_skills": ["Technical Strategy", "System Architecture", "Agile Leadership"] + cand_graph.skills[:3],
            "key_bullet_points": [
                "Led engineering initiatives from requirement discovery to production deployment, delivering key milestones ahead of schedule.",
                "Mentored engineering talent and established modern DevOps & continuous delivery standards.",
                "Optimized cloud resource allocation, driving significant operational cost savings."
            ]
        }

        state.resume_variants = {
            "ats_variant": ats_variant,
            "technical_variant": technical_variant,
            "executive_variant": executive_variant,
            "rag_citations": rag_context
        }

        return state
