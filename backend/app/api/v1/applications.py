from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Dict, Any
from app.db.database import get_db
from app.db.models import Application, JobPosting
from app.agents.content_agent import content_agent

router = APIRouter()

@router.get("/")
async def list_applications(db: AsyncSession = Depends(get_db)):
    """List candidate applications for Kanban tracking board."""
    result = await db.execute(select(Application))
    apps = result.scalars().all()
    
    response = []
    for app in apps:
        job = await db.get(JobPosting, app.job_id)
        response.append({
            "id": app.id,
            "job_id": app.job_id,
            "status": app.status, # Discovered, Shortlisted, Applied, Interviewing, Offer, Rejected
            "template_used": app.template_used,
            "package_assembled": app.package_assembled,
            "overleaf_url": app.overleaf_url,
            "job_title": job.title if job else "Software Engineer",
            "company": job.company if job else "Tech Company",
            "location": job.location if job else "Bengaluru",
            "salary_range": job.salary_range if job else "₹ 15-20 LPA",
            "updated_at": app.updated_at.strftime("%d %b %Y") if app.updated_at else "Recently"
        })
    return response

@router.post("/package/{job_id}")
async def generate_package(job_id: str, payload: Dict[str, Any] = Body(default={}), db: AsyncSession = Depends(get_db)):
    """Generate tailored resume, cover letter, Q&A, and Overleaf URL for a target job."""
    job = await db.get(JobPosting, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    template_id = payload.get("template_id", "jakes-resume")
    package = content_agent.generate_application_package(
        job_title=job.title,
        company=job.company,
        candidate_name="Arjun B.",
        template_id=template_id
    )
    
    # Check if application already exists
    app_res = await db.execute(select(Application).where(Application.job_id == job_id))
    app = app_res.scalar_one_or_none()
    
    if not app:
        app = Application(
            user_id=1,
            job_id=job_id,
            status="Shortlisted",
            template_used=template_id,
            cover_letter=package["cover_letter"],
            qa_answers=package["qa_answers"],
            overleaf_url=package["overleaf_url"],
            package_assembled=True
        )
        db.add(app)
    else:
        app.template_used = template_id
        app.cover_letter = package["cover_letter"]
        app.qa_answers = package["qa_answers"]
        app.overleaf_url = package["overleaf_url"]
        app.package_assembled = True
        
    await db.commit()
    await db.refresh(app)
    
    return {
        "application_id": app.id,
        "job_id": job_id,
        "package": package
    }

@router.patch("/{app_id}/status")
async def update_application_status(app_id: int, payload: Dict[str, Any] = Body(...), db: AsyncSession = Depends(get_db)):
    """Update Kanban status for an application (e.g. Applied, Interviewing, Offer)."""
    app = await db.get(Application, app_id)
    if not app:
        raise HTTPException(status_code=404, detail="Application not found")
        
    new_status = payload.get("status")
    if new_status:
        app.status = new_status
        await db.commit()
        await db.refresh(app)
        
    return {"id": app.id, "status": app.status}
