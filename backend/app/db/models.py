import datetime
from sqlalchemy import Column, Integer, String, Text, Float, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from app.db.session import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    reports = relationship("Report", back_populates="user")

class Resume(Base):
    __tablename__ = "resumes"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=True)
    raw_text = Column(Text, nullable=False)
    parsed_graph = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class JobDescription(Base):
    __tablename__ = "job_descriptions"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=True)
    company = Column(String, nullable=True)
    raw_text = Column(Text, nullable=False)
    parsed_data = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class Report(Base):
    __tablename__ = "reports"
    
    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(String, unique=True, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    match_score = Column(Float, nullable=False)
    ats_score = Column(Float, nullable=False)
    candidate_name = Column(String, nullable=True)
    target_role = Column(String, nullable=True)
    report_data = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    user = relationship("User", back_populates="reports")

class ModelUsageLog(Base):
    __tablename__ = "model_usage_log"
    
    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(String, index=True, nullable=False)
    agent_name = Column(String, nullable=False)
    tier = Column(Integer, nullable=False)
    model_name = Column(String, nullable=False)
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    estimated_cost_usd = Column(Float, default=0.0)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
