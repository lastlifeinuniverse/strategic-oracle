"""
Divination Reflection Generation Module
========================================
Generates psychologically-grounded, archetypal daily reflections.

Uses your existing LLM stack:
- DeepSeek via OpenRouter (primary — for psychological/archetypal reasoning)
- Gemini 2.5-flash (optional — for tone refinement)

Integrates with:
- OracleDB (life_context table for user state, divination_history for storage)
- strategic_oracle_gemini.OracleServices (Bazi meta + clients)

Output: 4-field structure (optimized for UX + token efficiency)
- headline: 1-sentence core pattern
- focus: 1-sentence actionable focus
- watch: 1-sentence mindfulness/trap
- full_reflection: 8-part detailed reading (lazy-loaded)

Usage:
    from divination_module import DivinationService
    svc = DivinationService(db=db)
    result = svc.generate_daily_reflection()
    # Returns: {success, headline, focus, watch, full_reflection, ...}
"""

import os
import json
import datetime
from typing import Optional, Dict, Any

DIVINATION_SYSTEM_PROMPT = """You are an insightful reflective guide skilled in:
- Jungian archetypes
- I Ching symbolism
- Tarot archetypes
- psychology
- personality pattern recognition
- life transitions and liminality
- coaching grounded in actual self-knowledge

You are NOT a fortune teller.
You do NOT predict supernatural outcomes.
You do NOT use clichés, generic positivity, or overdramatic spiritual language.

Your role is to generate a daily "divination-style reflection" based on:
1. The user's current life context (career transition, identity work)
2. Their personality patterns (high openness/agreeableness, neuroticism 71, natural facilitator)
3. Their current emotional/situational phase (liminality between banking and meaningful work)
4. Symbolic archetypes from systems like I Ching and Tarot (as metaphorical mirrors)
5. A light Ba Zi layer (background personality/timing influence only — NOT deterministic)

The user is a 43-year-old Bing Fire (丙 Yang Fire) Day Master navigating:
- Career transition from banking → teaching/coaching/meaningful AI work
- Identity shift from achievement-driven → meaning-driven
- Core tension: competence in banking vs. authenticity in purpose-driven work
- Life phase (2025-2035 luck cycle): self-amplification, identity-driven decisions
- Psychological profile: high openness (92), high agreeableness (90), neuroticism (71)
- Natural tendency: deep feeler, guides/facilitates, prone to anxiety under rigid systems

CRITICAL OUTPUT FORMAT:

You must respond with ONLY a valid JSON object (no markdown, no extra text):

{
  "headline": "One-sentence core pattern or insight for today",
  "focus": "One-sentence actionable focus — what to lean into or prioritize",
  "watch": "One-sentence mindfulness warning — the psychological trap or pattern to be aware of",
  "full_reflection": "Full 8-part divination reflection (see structure below)"
}

FULL REFLECTION STRUCTURE (for the full_reflection field):
1. **Core Theme** — The overarching insight for this day (2-3 sentences)
2. **Symbolic Reading**
   - *I Ching feel:* Which hexagram principle applies and why
   - *Tarot archetype:* Which major arcana mirrors this moment and why
3. **Emotional / Mental Tendencies** — What patterns are activated (3-4 sentences)
4. **Risks / Watch-outs** — Where you might overextend or miss the signal (2-3 sentences)
5. **Best Way to Move Today** — How to navigate skillfully (2-3 sentences)
6. **Aligned Actions** — 3-4 specific bullet points of practices/approaches that fit
7. **Hidden Opportunity** — What becomes possible if you stay awake (2-3 sentences)
8. **Closing Insight** — A grounding final thought (1-2 sentences)

Tone for all fields:
- calm, thoughtful, emotionally intelligent
- practical and poetic but grounded
- avoid mystical exaggeration, clichés, generic positivity
- speak directly to their life — not generically
- headline/focus/watch should be concrete and psychologically specific
- name actual patterns they're likely experiencing (not universal wisdom)

AVOID:
- "You are a natural leader" / "Trust the universe" / "Everything happens for a reason"
- "This is a time of abundance" / "Great things are coming"
- Vague statements like "follow your intuition" without context
- Overdramatic spiritual language ("divine timing", "cosmic alignment")
- Extreme certainty ("You will...", "This guarantees...")
- Generic horoscope-style predictions

EMBRACE:
- Name their specific psychological patterns (anxiety under pressure, tendency to over-give)
- Reference their actual life situation (identity transition, tension between authenticity and competence)
- Acknowledge the liminality/discomfort of their phase (not bypassing it with positivity)
- Suggest what to *notice* rather than what will *happen*
- Respect their intelligence and self-knowledge

Use the user context HEAVILY. Treat the symbolic systems as metaphorical mirrors,
not objective truth. The Ba Zi layer is background personality/timing influence, never destiny.
The goal is psychological insight grounded in their actual situation, not mystical reassurance."""


class DivinationService:
    """Generate psychologically-grounded daily reflections using DeepSeek/Gemini."""

    def __init__(self, db=None, model_choice: str = "deepseek"):
        """
        Args:
            db: OracleDB instance (for context fetching + reflection storage)
            model_choice: "deepseek" (default, via OpenRouter) or "gemini" (2.5-flash)
        """
        self.db = db
        self.model_choice = model_choice
        self._openrouter_client = None
        self._gemini_client = None
        self._deepseek_model = None
        self._gemini_model = None
        self._init_clients()

    def _init_clients(self):
        """Initialize clients by reusing strategic_oracle_gemini setup."""
        try:
            from strategic_oracle_gemini import (
                openrouter_client,
                gemini_client,
                REASONING_MODEL,
                GEMINI_MODEL,
            )
            self._openrouter_client = openrouter_client
            self._gemini_client = gemini_client
            self._deepseek_model = REASONING_MODEL
            self._gemini_model = GEMINI_MODEL
        except Exception as e:
            print(f"[warn] Could not load LLM clients: {e}")

    def fetch_user_context(self) -> Dict[str, Any]:
        """Fetch user's current life context from DB or fall back to defaults."""
        if self.db and hasattr(self.db, "get_current_life_context"):
            return self.db.get_current_life_context()
        return {
            "situation": "unknown",
            "emotional_state": "unknown",
            "focus_area": "unknown",
            "recent_events": "none",
            "phase": "unknown",
            "goals": "unknown"
        }

    def fetch_bazi_meta(self) -> Dict[str, Any]:
        """Fetch today's Bazi metadata via OracleServices."""
        try:
            from strategic_oracle_gemini import OracleServices
            svc = OracleServices()
            return svc.get_today_meta()
        except Exception as e:
            print(f"[warn] Could not fetch Bazi meta: {e}")
            return {
                "date": str(datetime.date.today()),
                "day_pillar": "unknown",
                "element_desc": "balanced",
                "officer_name": "Balance"
            }

    def build_user_message(self, user_context: Dict, bazi_meta: Dict) -> str:
        """Compose the user message including all context."""
        lines = [
            "Generate my daily divination-style reflection based on this context:",
            "",
            "=== TODAY'S BA ZI TIMING (background influence only) ===",
            f"Date: {bazi_meta.get('date', 'unknown')}",
            f"Day Pillar: {bazi_meta.get('day_pillar', 'unknown')}",
            f"Element Energy: {bazi_meta.get('element_desc', 'unknown')}",
            f"Branch Energy: {bazi_meta.get('branch_energy', '')}",
            f"Officer: {bazi_meta.get('officer_name', 'Balance')} — {bazi_meta.get('officer_meaning', '')}",
            f"Day Quality for my Bing Fire: {bazi_meta.get('day_quality', '')}",
            f"Personal Impact: {bazi_meta.get('personal_impact', '')}",
            "",
            "=== MY CURRENT LIFE CONTEXT ===",
            f"Life Situation: {user_context.get('situation', '—')}",
            f"Emotional State: {user_context.get('emotional_state', '—')}",
            f"Current Focus: {user_context.get('focus_area', '—')}",
            f"Recent Events: {user_context.get('recent_events', '—')}",
            f"Life Phase: {user_context.get('phase', '—')}",
            f"Current Goals: {user_context.get('goals', '—')}",
            "",
            "=== MY PROFILE PATTERNS ===",
            "Day Master: 丙 Bing (Yang Fire / Sun Fire) — warm, expressive, wants to guide",
            "Luck Cycle 2025-2035: self-amplification, identity-driven decisions, authenticity",
            "Personality: high openness (92), high agreeableness (90), neuroticism (71)",
            "Tendencies: deep feeler/thinker, prone to anxiety under rigid systems, natural facilitator",
            "Practice: yoga instructor, runner, SCA coffee brewer, AI student",
            "Direction: transitioning from banking toward teaching/coaching/meaningful AI work",
            "",
            "Now create my divination reflection as JSON with headline, focus, watch, and full_reflection fields.",
        ]
        return "\n".join(lines)

    def _parse_json_response(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract and parse JSON from LLM response."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except:
                    return None
        return None

    def _call_deepseek(self, system: str, user_message: str) -> str:
        """Call DeepSeek via OpenRouter."""
        if not self._openrouter_client:
            raise RuntimeError("OpenRouter client not initialized")
        response = self._openrouter_client.chat.completions.create(
            model=self._deepseek_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=2400,
        )
        return response.choices[0].message.content.strip()

    def _call_gemini(self, system: str, user_message: str) -> str:
        """Call Gemini 2.5-flash."""
        if not self._gemini_client:
            raise RuntimeError("Gemini client not initialized")
        full_prompt = f"{system}\n\n---\n\n{user_message}"
        response = self._gemini_client.models.generate_content(
            model=self._gemini_model,
            contents=full_prompt,
        )
        return response.text.strip()

    def generate_daily_reflection(
        self,
        user_context: Optional[Dict] = None,
        save: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a full daily divination reflection with 4 fields:
        - headline: 1-sentence core pattern
        - focus: 1-sentence actionable focus
        - watch: 1-sentence mindfulness warning
        - full_reflection: 8-part detailed reading

        Args:
            user_context: Optional override (otherwise fetched from DB)
            save: Whether to persist to divination_history table

        Returns dict with: success, headline, focus, watch, full_reflection, etc.
        """
        if user_context is None:
            user_context = self.fetch_user_context()

        bazi_meta = self.fetch_bazi_meta()
        user_message = self.build_user_message(user_context, bazi_meta)

        try:
            if self.model_choice == "gemini":
                response_text = self._call_gemini(DIVINATION_SYSTEM_PROMPT, user_message)
                model_used = self._gemini_model or "gemini-2.5-flash"
            else:
                response_text = self._call_deepseek(DIVINATION_SYSTEM_PROMPT, user_message)
                model_used = self._deepseek_model or "deepseek/deepseek-chat"

            # Parse JSON response
            parsed = self._parse_json_response(response_text)
            if not parsed:
                return {
                    "success": False,
                    "error": "Could not parse LLM response as JSON",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "model": self.model_choice
                }

            result = {
                "success": True,
                "headline": parsed.get("headline", ""),
                "focus": parsed.get("focus", ""),
                "watch": parsed.get("watch", ""),
                "full_reflection": parsed.get("full_reflection", ""),
                "timestamp": datetime.datetime.now().isoformat(),
                "model": model_used,
                "user_context": user_context,
                "bazi_meta": {
                    "date": bazi_meta.get("date"),
                    "day_pillar": bazi_meta.get("day_pillar"),
                    "element": bazi_meta.get("element_desc"),
                    "officer": bazi_meta.get("officer_name")
                }
            }

            # Persist to DB
            if save and self.db and hasattr(self.db, "save_divination_reflection"):
                try:
                    reflection_id = self.db.save_divination_reflection({
                        "day_pillar": bazi_meta.get("day_pillar", ""),
                        "element": bazi_meta.get("element_desc", ""),
                        "officer": bazi_meta.get("officer_name", ""),
                        "user_context": user_context,
                        "headline": parsed.get("headline", ""),
                        "focus": parsed.get("focus", ""),
                        "watch": parsed.get("watch", ""),
                        "reflection_text": parsed.get("full_reflection", ""),
                        "model_used": model_used
                    })
                    result["reflection_id"] = reflection_id
                except Exception as e:
                    print(f"[warn] Could not save reflection: {e}")

            return result

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.datetime.now().isoformat(),
                "model": self.model_choice
            }

    def get_or_generate_today(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Return today's cached reflection if it exists, otherwise generate fresh."""
        if not force_refresh and self.db and hasattr(self.db, "get_today_divination"):
            cached = self.db.get_today_divination()
            if cached:
                return {
                    "success": True,
                    "headline": cached.get("headline", ""),
                    "focus": cached.get("focus", ""),
                    "watch": cached.get("watch", ""),
                    "full_reflection": cached.get("reflection_text", ""),
                    "timestamp": cached.get("created_at", ""),
                    "model": cached.get("model_used", "cached"),
                    "from_cache": True,
                    "bazi_meta": {
                        "date": cached.get("reflection_date"),
                        "day_pillar": cached.get("day_pillar"),
                        "element": cached.get("element"),
                        "officer": cached.get("officer")
                    }
                }

        result = self.generate_daily_reflection()
        result["from_cache"] = False
        return result


if __name__ == "__main__":
    svc = DivinationService(db=None)
    print(f"DeepSeek model: {svc._deepseek_model}")
    print(f"OpenRouter ready: {svc._openrouter_client is not None}\n")

    if svc._openrouter_client:
        result = svc.generate_daily_reflection(save=False)
        if result["success"]:
            print("=" * 80)
            print(f"HEADLINE: {result['headline']}\n")
            print(f"FOCUS: {result['focus']}\n")
            print(f"WATCH: {result['watch']}\n")
            print("=" * 80)
            print("FULL REFLECTION:")
            print(result['full_reflection'])
            print("=" * 80)
        else:
            print(f"[ERROR] {result['error']}")
