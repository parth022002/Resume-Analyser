from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.db.models import Report

router = APIRouter()

@router.get("/reports/{report_id}")
def get_report(report_id: str, db: Session = Depends(get_db)):
    report = db.query(Report).filter(Report.report_id == report_id).first()
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report with ID '{report_id}' not found."
        )
    return {
        "status": "success",
        "data": report.report_data
    }

@router.get("/reports/{report_id}/export", response_class=HTMLResponse)
def export_report_html(report_id: str, db: Session = Depends(get_db)):
    """Generate printable HTML report view for PDF export."""
    report = db.query(Report).filter(Report.report_id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
        
    data = report.report_data
    meta = data.get("metadata", {})
    scores = data.get("scores", {})
    skills = data.get("skills_analysis", {})
    action_plan = data.get("action_plan", [])
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>TalentForge Report - {meta.get('candidate_name', 'Candidate')}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; color: #1e293b; background: #fff; }}
        .header {{ border-bottom: 2px solid #0284c7; padding-bottom: 15px; margin-bottom: 20px; }}
        .title {{ font-size: 24px; font-weight: bold; color: #0f172a; }}
        .badge {{ background: #e0f2fe; color: #0369a1; padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: bold; }}
        .grid {{ display: flex; gap: 20px; margin: 20px 0; }}
        .card {{ flex: 1; border: 1px solid #cbd5e1; padding: 15px; border-radius: 8px; text-align: center; }}
        .score {{ font-size: 32px; font-weight: bold; color: #0284c7; }}
        .section {{ margin-top: 25px; }}
        .skill-tag {{ display: inline-block; background: #f1f5f9; padding: 4px 8px; border-radius: 4px; margin: 2px; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="header">
        <span class="badge">TalentForge v2.0 AI Career Intelligence</span>
        <div class="title">{meta.get('candidate_name', 'Candidate')}</div>
        <div>Target Role: {meta.get('target_role', 'Software Engineer')} | Company: {meta.get('target_company', 'Target Company')}</div>
    </div>
    
    <div class="grid">
        <div class="card">
            <div>Overall Match</div>
            <div class="score">{scores.get('overall_match_score', 80)}%</div>
        </div>
        <div class="card">
            <div>ATS Readability</div>
            <div class="score">{scores.get('ats_compatibility_score', 85)}%</div>
        </div>
        <div class="card">
            <div>Interview Readiness</div>
            <div class="score">{scores.get('interview_readiness_score', 88)}%</div>
        </div>
    </div>

    <div class="section">
        <h3>Matched Skills</h3>
        <div>{" ".join([f'<span class="skill-tag">{s}</span>' for s in skills.get('matching_skills', [])])}</div>
    </div>

    <div class="section">
        <h3>Action Plan</h3>
        <ul>
            {"".join([f'<li>{a}</li>' for a in action_plan])}
        </ul>
    </div>
</body>
</html>"""
    return html_content

@router.get("/reports")
def list_reports(limit: int = 10, db: Session = Depends(get_db)):
    reports = db.query(Report).order_by(Report.created_at.desc()).limit(limit).all()
    return {
        "status": "success",
        "count": len(reports),
        "data": [
            {
                "report_id": r.report_id,
                "candidate_name": r.candidate_name,
                "target_role": r.target_role,
                "match_score": r.match_score,
                "ats_score": r.ats_score,
                "created_at": r.created_at
            }
            for r in reports
        ]
    }
