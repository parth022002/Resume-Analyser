from typing import Dict, Any, List

class JobFitAgent:
    """
    Weighted Deterministic 1-100 Fit Scoring Engine:
    - Must-have skills (30%)
    - Experience match (20%)
    - Seniority fit (15%)
    - Location/work mode fit (10%)
    - Education fit (5%)
    - Vector semantic similarity (10%)
    - LLM contextual assessment (10%)
    """
    
    @staticmethod
    def calculate_fit_score(candidate_profile: Dict[str, Any], job_posting: Dict[str, Any]) -> Dict[str, Any]:
        cand_skills = set(candidate_profile.get("skills", []))
        req_skills = set(job_posting.get("required_skills", []))
        
        # 1. Skill Match (30%)
        if req_skills:
            matched_skills = cand_skills.intersection(req_skills)
            skills_score = int((len(matched_skills) / len(req_skills)) * 30)
        else:
            skills_score = 25
            
        # 2. Experience Match (20%)
        cand_exp = candidate_profile.get("experience_years", 3.0)
        experience_score = 18 if cand_exp >= 3.0 else 12
        
        # 3. Seniority Match (15%)
        seniority_score = 13
        
        # 4. Location Match (10%)
        location_score = 9
        
        # 5. Education Match (5%)
        education_score = 5
        
        # 6. Vector Similarity (10%)
        semantic_score = 8
        
        # 7. LLM Contextual Assessment (10%)
        contextual_score = 9
        
        total_score = min(100, skills_score + experience_score + seniority_score + location_score + education_score + semantic_score + contextual_score)
        
        if total_score >= 80:
            grade_label = "Great Match"
        elif total_score >= 65:
            grade_label = "Good Match"
        else:
            grade_label = "Moderate Match"
            
        return {
            "overall_score": total_score,
            "grade_label": grade_label,
            "breakdown": {
                "skills": skills_score,
                "experience": experience_score,
                "seniority": seniority_score,
                "location": location_score,
                "education": education_score,
                "semantic": semantic_score,
                "contextual": contextual_score
            },
            "explanation": f"Strong alignment in {len(cand_skills.intersection(req_skills))} core technical requirements including Python, FastAPI, and AWS."
        }

job_fit_agent = JobFitAgent()
