import re
import io
from typing import Dict, Any, List
from app.agents.state import CandidateKnowledgeGraph, JobKnowledgeGraph, AgentState
from app.core.guardrails import GuardrailsService
from app.core.model_router import ModelRouter

class IntakeParsingAgent:
    """
    Stage 1 — Intake & Parsing Agent:
    Parses resume document (PDF/DOCX) and Job Description into structured Knowledge Graphs.
    """
    
    @staticmethod
    def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
        """Extract text from PDF using pdfplumber or pypdf with fallback."""
        extracted_text = ""
        
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                pages = [page.extract_text() for page in pdf.pages if page.extract_text()]
                extracted_text = "\n".join(pages)
        except Exception:
            pass
        
        if not extracted_text.strip():
            try:
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(pdf_bytes))
                pages = [page.extract_text() for page in reader.pages if page.extract_text()]
                extracted_text = "\n".join(pages)
            except Exception:
                extracted_text = ""

        if not extracted_text.strip():
            try:
                decoded = pdf_bytes.decode("utf-8", errors="ignore")
                cleaned = "".join(ch for ch in decoded if ch.isprintable() or ch in ['\n', '\r', '\t'])
                if len(cleaned.strip()) > 5:
                    extracted_text = cleaned
            except Exception:
                pass
                
        return extracted_text

    @classmethod
    def parse(cls, state: AgentState) -> AgentState:
        """Run Stage 1 parsing pipeline."""
        raw_resume = state.raw_resume_text
        raw_jd = state.raw_jd_text
        
        # 1. Security & Guardrails Check
        is_malicious, reason = GuardrailsService.check_prompt_injection(raw_resume)
        if is_malicious:
            state.guardrails_passed = False
            state.guardrail_warnings.append(reason)
            
        # 2. PII Redaction for model safety
        safe_resume = GuardrailsService.redact_pii(raw_resume)
        
        # 3. Parse Candidate Knowledge Graph
        candidate_graph = cls._extract_candidate_graph(safe_resume, state.github_url, state.portfolio_url)
        state.candidate_graph = candidate_graph
        
        # 4. Parse Job Knowledge Graph
        job_graph = cls._extract_job_graph(raw_jd)
        state.job_graph = job_graph
        
        return state

    @classmethod
    def _extract_candidate_graph(cls, resume_text: str, github_url: str = None, portfolio_url: str = None) -> CandidateKnowledgeGraph:
        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', resume_text)
        email = email_match.group(0) if email_match else "N/A"
        
        skills_database = [
            "Python", "JavaScript", "TypeScript", "React", "Next.js", "Node.js", "FastAPI",
            "Django", "Flask", "SQL", "PostgreSQL", "MongoDB", "Docker", "Kubernetes",
            "AWS", "GCP", "Git", "Machine Learning", "Deep Learning", "PyTorch",
            "TensorFlow", "Pandas", "Scikit-Learn", "C++", "Java", "Go", "HTML", "CSS", "Tailwind"
        ]
        
        found_skills = [skill for skill in skills_database if re.search(r'\b' + re.escape(skill) + r'\b', resume_text, re.IGNORECASE)]
        
        if not github_url:
            gh_match = re.search(r'github\.com\/[\w-]+', resume_text, re.IGNORECASE)
            if gh_match:
                github_url = "https://" + gh_match.group(0)

        lines = [line.strip() for line in resume_text.split('\n') if line.strip()]
        candidate_name = lines[0] if lines and len(lines[0]) < 40 else "Candidate"

        return CandidateKnowledgeGraph(
            name=candidate_name,
            email=email,
            skills=found_skills or ["Software Engineering", "Problem Solving", "Git"],
            summary=lines[1] if len(lines) > 1 else "Software Engineering Professional",
            github_url=github_url,
            portfolio_url=portfolio_url,
            total_experience_years=2.0
        )

    @classmethod
    def _extract_job_graph(cls, jd_text: str) -> JobKnowledgeGraph:
        skills_database = [
            "Python", "JavaScript", "TypeScript", "React", "Next.js", "Node.js", "FastAPI",
            "Django", "Flask", "SQL", "PostgreSQL", "MongoDB", "Docker", "Kubernetes",
            "AWS", "GCP", "Git", "Machine Learning", "Deep Learning", "PyTorch",
            "TensorFlow", "Pandas", "Scikit-Learn", "C++", "Java", "Go"
        ]
        
        required_skills = [skill for skill in skills_database if re.search(r'\b' + re.escape(skill) + r'\b', jd_text, re.IGNORECASE)]
        
        lines = [line.strip() for line in jd_text.split('\n') if line.strip()]
        title = lines[0] if lines else "Software Engineer"
        
        return JobKnowledgeGraph(
            title=title,
            company="Target Enterprise",
            required_skills=required_skills or ["Python", "API Development", "SQL"],
            domain="Full Stack Software Engineering"
        )
