import logging
import datetime
from typing import Dict, Any
from app.db.session import SessionLocal, engine
from app.db.models import ModelUsageLog, Base

logger = logging.getLogger("langfuse_monitoring")

class MonitoringService:
    """
    Langfuse & Token Cost Observability Layer.
    Logs LLM call metadata, model tiers, token usage, and estimated USD costs.
    """
    
    TIER_COSTS = {
        0: 0.0000,   # Tier 0 (Free Gemini Flash / Llama 3.3)
        1: 0.0002,   # Tier 1 (DeepSeek V3 / Qwen)
        2: 0.0030    # Tier 2 (Claude Sonnet / GPT-4.1)
    }

    @classmethod
    def log_agent_call(
        cls, 
        report_id: str, 
        agent_name: str, 
        tier: int, 
        model_name: str, 
        prompt_tokens: int = 450, 
        completion_tokens: int = 250
    ):
        cost_per_1k = cls.TIER_COSTS.get(tier, 0.0005)
        total_tokens = prompt_tokens + completion_tokens
        estimated_cost = round((total_tokens / 1000.0) * cost_per_1k, 6)

        try:
            # Auto-create tables if not present
            Base.metadata.create_all(bind=engine)
            
            db = SessionLocal()
            log_entry = ModelUsageLog(
                report_id=report_id,
                agent_name=agent_name,
                tier=tier,
                model_name=model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                estimated_cost_usd=estimated_cost,
                timestamp=datetime.datetime.utcnow()
            )
            db.add(log_entry)
            db.commit()
            db.close()
        except Exception as e:
            logger.warning(f"Could not record model usage log to DB ({e}).")
