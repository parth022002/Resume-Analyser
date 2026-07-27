import logging
import json
from typing import Dict, Any, Optional
from app.core.config import settings

logger = logging.getLogger("model_router")

class ModelRouter:
    """
    Enhanced Multi-Model Router using OpenRouter, Gemini, and Groq:
    - Tier 0: Fast / Free Intake & Security Guardrails (Gemini 2.5 Flash / NVIDIA Nemotron Safety)
    - Tier 1: Matching, Code Review & Skill Insights (DeepSeek-V3 / Google Gemma / Groq Llama 3.3)
    - Tier 2: Resume Optimization & Strategic Career Guidance (NVIDIA Nemotron / DeepSeek R1 / Claude)
    """
    
    TIER_0 = 0 # Intake & Safety Guardrails
    TIER_1 = 1 # Match Scoring & Code Review (DeepSeek / Gemma)
    TIER_2 = 2 # Strategic Career Synthesis (NVIDIA Nemotron / DeepSeek R1)

    MODEL_MAP = {
        0: settings.TIER_0_MODEL,
        1: settings.TIER_1_MODEL,
        2: settings.TIER_2_MODEL,
        "safety": "openrouter/nvidia/nemotron-4-340b-instruct",
        "code_review": "openrouter/deepseek/deepseek-chat",
        "skill_insight": "openrouter/google/gemma-2-27b-it",
        "career_strategy": "openrouter/nvidia/nemotron-4-340b-instruct"
    }
    
    @classmethod
    def get_model_name(cls, tier: Any) -> str:
        if isinstance(tier, int):
            return cls.MODEL_MAP.get(tier, settings.TIER_0_MODEL)
        return cls.MODEL_MAP.get(tier, settings.TIER_1_MODEL)

    @classmethod
    def generate_completion(
        cls, 
        tier: Any, 
        system_prompt: str, 
        user_prompt: str, 
        fallback_json: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes an LLM call via LiteLLM/OpenRouter with dynamic model selection.
        Falls back seamlessly to the rule-based parser if API keys or network calls fail.
        """
        model_name = cls.get_model_name(tier)
        active_key = (
            settings.OPENROUTER_API_KEY or 
            settings.GEMINI_API_KEY or 
            settings.GROQ_API_KEY or 
            settings.OPENAI_API_KEY
        )
        
        # Check if API keys exist
        if not active_key:
            logger.warning(f"No active API key set. Using rule-based fallback parser for task '{tier}'.")
            if fallback_json is not None:
                return fallback_json
            return {"status": "success", "note": "Rule-based fallback active."}

        try:
            import litellm
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            response = litellm.completion(
                model=model_name,
                messages=messages,
                api_key=active_key,
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            logger.error(f"LiteLLM invocation failed for model {model_name}: {str(e)}. Using fallback engine.")
            if fallback_json is not None:
                return fallback_json
            return {"error": str(e), "fallback": True}
