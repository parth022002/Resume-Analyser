import datetime
from sqlalchemy import Column, Integer, String, Text, Float, Boolean, DateTime, ForeignKey, JSON
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String(255), default="Arjun B.")
    email = Column(String(255), unique=True, index=True, default="arjun@talentforge.ai")
    password_hash = Column(String(255), default="hashed_demo_password")
    avatar_url = Column(String(512), default="https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=150")
    plan = Column(String(50), default="Free Student Account")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    profiles = relationship("CandidateProfile", back_populates="user")
    applications = relationship("Application", back_populates="user")
    target_companies = relationship("TargetCompany", back_populates="user")


class CandidateProfile(Base):
    __tablename__ = "candidate_profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    headline = Column(String(255), default="Software Engineer - Backend & Systems")
    skills = Column(JSON, default=["Python", "FastAPI", "AWS", "PostgreSQL", "Docker", "CI/CD", "REST APIs", "Microservices", "System Design"])
    experience_years = Column(Float, default=3.5)
    education = Column(String(255), default="B.Tech Computer Science")
    preferred_roles = Column(JSON, default=["Backend Developer", "Software Engineer", "Full Stack Developer", "SDE II"])
    preferred_locations = Column(JSON, default=["Bengaluru", "Remote", "Hybrid"])
    min_salary_lpa = Column(Float, default=15.0)
    resume_text = Column(Text, nullable=True)
    portfolio_url = Column(String(512), default="https://arjun.dev")
    github_url = Column(String(512), default="https://github.com/arjun-b")
    linkedin_url = Column(String(512), default="https://linkedin.com/in/arjun-b")
    leetcode_url = Column(String(512), default="https://leetcode.com/arjun_b")
    other_urls = Column(String(512), default="https://twitter.com/arjun_dev")
    education_details = Column(JSON, default=[
        {"degree": "B.Tech Computer Science", "institute": "RV College of Engineering", "year": "2024", "cgpa": "8.8/10"}
    ])
    certifications = Column(JSON, default=[
        {"title": "AWS Certified Developer – Associate", "issuer": "Amazon Web Services", "year": "2023"},
        {"title": "PostgreSQL Professional Certification", "issuer": "PostgreSQL Institute", "year": "2023"}
    ])
    extracurricular = Column(JSON, default=[
        {"title": "Hackathon Winner", "org": "Smart India Hackathon", "desc": "1st place out of 500+ teams building real-time logistics routing"},
        {"title": "Open Source Contributor", "org": "FastAPI Ecosystem", "desc": "Contributed performance benchmarks and documentation fixes"}
    ])
    resume_score = Column(Integer, default=88)
    knowledge_graph = Column(JSON, default={})
    updated_at = Column(DateTime, default=datetime.datetime.utcnow)

    user = relationship("User", back_populates="profiles")


class JobPosting(Base):
    __tablename__ = "job_postings"

    id = Column(String(255), primary_key=True) # e.g. "job-101"
    title = Column(String(255), index=True)
    company = Column(String(255), index=True)
    logo_url = Column(String(512), nullable=True)
    location = Column(String(255))
    work_mode = Column(String(50), default="Hybrid") # Remote, On-site, Hybrid
    salary_range = Column(String(100), default="₹ 15 - 22 LPA")
    description = Column(Text)
    required_skills = Column(JSON, default=[])
    nice_to_have_skills = Column(JSON, default=[])
    source_type = Column(String(50), default="Target Company") # Target Company, Broad Search, Portal Alert, Simplify
    source_platform = Column(String(100), default="LinkedIn") # LinkedIn, Naukri.com, Greenhouse, Lever, Ashby, Indeed
    source_url = Column(String(512), nullable=True)
    posted_date = Column(String(50), default="2 days ago")
    is_target_company = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    fit_scores = relationship("FitScore", back_populates="job_posting")
    applications = relationship("Application", back_populates="job_posting")


class FitScore(Base):
    __tablename__ = "fit_scores"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(255), ForeignKey("job_postings.id"))
    overall_score = Column(Integer, default=85) # 1-100
    grade_label = Column(String(50), default="Great Match")
    skills_score = Column(Integer, default=30)
    experience_score = Column(Integer, default=18)
    seniority_score = Column(Integer, default=13)
    location_score = Column(Integer, default=9)
    education_score = Column(Integer, default=5)
    semantic_score = Column(Integer, default=8)
    contextual_score = Column(Integer, default=9)
    breakdown_json = Column(JSON, default={})
    explanation = Column(Text, default="Strong candidate alignment in core requirements.")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    job_posting = relationship("JobPosting", back_populates="fit_scores")


class Application(Base):
    __tablename__ = "applications"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    job_id = Column(String(255), ForeignKey("job_postings.id"))
    status = Column(String(50), default="Shortlisted") # Discovered, Shortlisted, Applied, Interviewing, Offer, Rejected
    template_used = Column(String(100), default="Jake's Resume (ATS Optimized)")
    cover_letter = Column(Text, nullable=True)
    qa_answers = Column(JSON, default={})
    overleaf_url = Column(String(512), nullable=True)
    package_assembled = Column(Boolean, default=True)
    applied_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow)

    user = relationship("User", back_populates="applications")
    job_posting = relationship("JobPosting", back_populates="applications")


class TargetCompany(Base):
    __tablename__ = "target_companies"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    company_name = Column(String(255), index=True)
    resolved_ats = Column(String(50), default="greenhouse") # greenhouse, lever, ashby, unresolved
    board_slug = Column(String(255), nullable=True)
    resolution_status = Column(String(50), default="resolved") # resolved, unresolved
    open_jobs_count = Column(Integer, default=12)
    last_polled = Column(String(50), default="10 mins ago")

    user = relationship("User", back_populates="target_companies")


class ResumeTemplate(Base):
    __tablename__ = "resume_templates"

    id = Column(String(100), primary_key=True) # e.g. "jakes-resume"
    name = Column(String(255))
    category = Column(String(100), default="ATS Safe / Tech")
    description = Column(Text)
    is_default = Column(Boolean, default=False)
    latex_code = Column(Text)


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"

    id = Column(Integer, primary_key=True, index=True)
    eval_type = Column(String(50)) # 'ranking' | 'fit_score' | 'content_quality' | 'time_efficiency'
    sample_size = Column(Integer, default=50)
    metric_name = Column(String(100))
    metric_value = Column(Float)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
