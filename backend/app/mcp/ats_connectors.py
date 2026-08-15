import httpx
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class ATSConnectors:
    @staticmethod
    async def fetch_greenhouse_jobs(board_token: str) -> List[Dict[str, Any]]:
        """Fetch public job listings from Greenhouse ATS."""
        url = f"https://boards-api.greenhouse.io/v1/boards/{board_token}/jobs?content=true"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.get(url)
                if res.status_code == 200:
                    data = res.json()
                    jobs = []
                    for job in data.get("jobs", []):
                        jobs.append({
                            "id": f"gh-{job.get('id')}",
                            "title": job.get("title"),
                            "company": board_token.capitalize(),
                            "location": job.get("location", {}).get("name", "Remote"),
                            "url": job.get("absolute_url"),
                            "description": job.get("content", ""),
                            "source_type": "Greenhouse ATS"
                        })
                    return jobs
        except Exception as e:
            logger.error(f"Error fetching Greenhouse board {board_token}: {e}")
        return []

    @staticmethod
    async def fetch_lever_jobs(company_slug: str) -> List[Dict[str, Any]]:
        """Fetch public job listings from Lever ATS."""
        url = f"https://api.lever.co/v0/postings/{company_slug}?mode=json"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.get(url)
                if res.status_code == 200:
                    data = res.json()
                    jobs = []
                    for job in data:
                        jobs.append({
                            "id": f"lever-{job.get('id')}",
                            "title": job.get("text"),
                            "company": company_slug.capitalize(),
                            "location": job.get("categories", {}).get("location", "Remote"),
                            "url": job.get("hostedUrl"),
                            "description": job.get("descriptionPlain", ""),
                            "source_type": "Lever ATS"
                        })
                    return jobs
        except Exception as e:
            logger.error(f"Error fetching Lever board {company_slug}: {e}")
        return []

    @staticmethod
    async def fetch_simplify_jobs() -> List[Dict[str, Any]]:
        """Fetch latest internship & new-grad roles from Simplify Jobs GitHub repository."""
        # Simulated Simplify Feed response
        return [
            {
                "id": "simp-1",
                "title": "Software Engineering Intern - Summer 2027",
                "company": "Stripe",
                "location": "Bengaluru / Remote",
                "url": "https://simplify.jobs/p/stripe-intern",
                "source_type": "Simplify Jobs Feed"
            },
            {
                "id": "simp-2",
                "title": "New Grad Software Engineer (2026)",
                "company": "Databricks",
                "location": "Bengaluru, KA",
                "url": "https://simplify.jobs/p/databricks-newgrad",
                "source_type": "Simplify Jobs Feed"
            }
        ]

ats_connectors = ATSConnectors()
