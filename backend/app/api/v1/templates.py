from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.db.database import get_db
from app.db.models import ResumeTemplate

router = APIRouter()

@router.get("/")
async def list_resume_templates(db: AsyncSession = Depends(get_db)):
    """List available LaTeX resume templates and Overleaf integration support."""
    result = await db.execute(select(ResumeTemplate))
    templates = result.scalars().all()
    
    return [
        {
            "id": t.id,
            "name": t.name,
            "category": t.category,
            "description": t.description,
            "is_default": t.is_default
        }
        for t in templates
    ]
