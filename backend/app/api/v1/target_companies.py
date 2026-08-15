from fastapi import APIRouter, Depends, HTTPException, Body, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Dict, Any, Optional
from app.db.database import get_db
from app.db.models import TargetCompany

router = APIRouter()

@router.get("/")
async def list_target_companies(user_id: Optional[int] = Query(1), db: AsyncSession = Depends(get_db)):
    """List monitored target companies for a specific student user."""
    stmt = select(TargetCompany).where(TargetCompany.user_id == user_id)
    result = await db.execute(stmt)
    companies = result.scalars().all()
    
    # If student has no target companies yet, seed initial default companies for user
    if len(companies) == 0:
        default_seeds = [
            {"company_name": "Google", "resolved_ats": "greenhouse", "open_jobs_count": 12},
            {"company_name": "Microsoft", "resolved_ats": "lever", "open_jobs_count": 8},
            {"company_name": "Amazon", "resolved_ats": "ashby", "open_jobs_count": 15},
            {"company_name": "Razorpay", "resolved_ats": "greenhouse", "open_jobs_count": 6}
        ]
        for ds in default_seeds:
            tc = TargetCompany(
                user_id=user_id,
                company_name=ds["company_name"],
                resolved_ats=ds["resolved_ats"],
                board_slug=ds["company_name"].lower(),
                resolution_status="resolved",
                open_jobs_count=ds["open_jobs_count"],
                last_polled="10 mins ago"
            )
            db.add(tc)
        await db.commit()
        
        stmt = select(TargetCompany).where(TargetCompany.user_id == user_id)
        result = await db.execute(stmt)
        companies = result.scalars().all()
    
    return [
        {
            "id": tc.id,
            "user_id": tc.user_id,
            "company_name": tc.company_name,
            "resolved_ats": tc.resolved_ats,
            "board_slug": tc.board_slug or tc.company_name.lower(),
            "resolution_status": tc.resolution_status,
            "open_jobs_count": tc.open_jobs_count,
            "last_polled": tc.last_polled
        }
        for tc in companies
    ]

@router.post("/")
async def add_target_company(payload: Dict[str, Any] = Body(...), db: AsyncSession = Depends(get_db)):
    """Add a new target company for a specific student user in Neon PostgreSQL."""
    company_name = payload.get("company_name")
    user_id = payload.get("user_id", 1)
    
    if not company_name or not company_name.strip():
        raise HTTPException(status_code=400, detail="Company name required")
        
    company_name = company_name.strip()
    slug = company_name.lower().replace(" ", "")
    
    # Determine ATS platform type
    ats_platform = "greenhouse"
    if "lever" in slug:
        ats_platform = "lever"
    elif "ashby" in slug:
        ats_platform = "ashby"
    elif "workday" in slug:
        ats_platform = "workday"
    
    tc = TargetCompany(
        user_id=user_id,
        company_name=company_name,
        resolved_ats=ats_platform,
        board_slug=slug,
        resolution_status="resolved",
        open_jobs_count=10,
        last_polled="Just now"
    )
    db.add(tc)
    await db.commit()
    await db.refresh(tc)
    
    return {
        "id": tc.id,
        "user_id": tc.user_id,
        "company_name": tc.company_name,
        "resolved_ats": tc.resolved_ats,
        "board_slug": tc.board_slug,
        "resolution_status": tc.resolution_status,
        "open_jobs_count": tc.open_jobs_count,
        "last_polled": tc.last_polled
    }

@router.delete("/{company_id}")
async def remove_target_company(company_id: int, user_id: Optional[int] = Query(1), db: AsyncSession = Depends(get_db)):
    """Remove a target company from the student's watchlist in Neon PostgreSQL."""
    tc = await db.get(TargetCompany, company_id)
    if not tc:
        raise HTTPException(status_code=404, detail="Target company record not found")
        
    await db.delete(tc)
    await db.commit()
    
    return {
        "status": "success",
        "message": f"Target company {company_id} removed from student watchlist.",
        "id": company_id
    }
