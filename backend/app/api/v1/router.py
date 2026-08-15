from fastapi import APIRouter
from app.api.v1 import jobs, applications, target_companies, templates, chat, analytics, auth, research

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(jobs.router, prefix="/jobs", tags=["jobs"])
api_router.include_router(applications.router, prefix="/applications", tags=["applications"])
api_router.include_router(target_companies.router, prefix="/target-companies", tags=["target_companies"])
api_router.include_router(templates.router, prefix="/templates", tags=["templates"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(analytics.router, prefix="/analytics", tags=["analytics"])
api_router.include_router(research.router, prefix="/research", tags=["research"])
