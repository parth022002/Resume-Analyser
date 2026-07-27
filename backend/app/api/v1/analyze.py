from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Optional
from app.db.session import get_db
from app.db.models import Report
from app.agents.stage1_intake.intake_parser import IntakeParsingAgent
from app.agents.graph import execute_career_pipeline

router = APIRouter()

@router.post("/analyze")
async def analyze_resume(
    resume_file: UploadFile = File(...),
    job_description: Optional[str] = Form(None),
    jd_file: Optional[UploadFile] = File(None),
    github_url: Optional[str] = Form(None),
    portfolio_url: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Primary API Endpoint: Accepts candidate resume PDF & target Job Description (text or uploaded PDF/TXT file).
    Runs the 4-Stage LangGraph Multi-Agent Pipeline.
    """
    if not resume_file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF resume uploads are currently supported."
        )

    # Extract Job Description text (either from text input or uploaded file)
    raw_jd_text = job_description or ""
    if jd_file and jd_file.filename:
        jd_contents = await jd_file.read()
        if jd_file.filename.endswith(".pdf"):
            extracted_jd = IntakeParsingAgent.extract_text_from_pdf_bytes(jd_contents)
            if extracted_jd.strip():
                raw_jd_text = extracted_jd
        else:
            try:
                raw_jd_text = jd_contents.decode("utf-8", errors="ignore")
            except Exception:
                pass

    if not raw_jd_text.strip():
        raw_jd_text = "Senior Software Engineer position requiring Python, REST APIs, SQL, and System Architecture."

    try:
        contents = await resume_file.read()
        raw_resume_text = IntakeParsingAgent.extract_text_from_pdf_bytes(contents)
        
        if not raw_resume_text.strip():
            raw_resume_text = f"Resume filename: {resume_file.filename}. Software Engineer candidate with background in software development."
            
        # Execute LangGraph Pipeline
        final_report = execute_career_pipeline(
            raw_resume_text=raw_resume_text,
            raw_jd_text=raw_jd_text,
            github_url=github_url,
            portfolio_url=portfolio_url
        )
        
        # Save Report to DB
        db_report = Report(
            report_id=final_report["report_id"],
            match_score=final_report["scores"]["overall_match_score"],
            ats_score=final_report["scores"]["ats_compatibility_score"],
            candidate_name=final_report["metadata"]["candidate_name"],
            target_role=final_report["metadata"]["target_role"],
            report_data=final_report
        )
        db.add(db_report)
        db.commit()
        db.refresh(db_report)
        
        return {
            "status": "success",
            "message": "Career Intelligence Analysis completed.",
            "data": final_report
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis pipeline error: {str(e)}"
        )
