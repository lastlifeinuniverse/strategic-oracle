"""
Strategic Oracle  LangGraph Controller with SQLite Memory & Optimized Performance
====================================================================================
OPTIMIZED IMPLEMENTATION:
- Ultra-fast local classification and keyword extraction (0.001s)
- Profile-aware keyword generation with 8 extracted traits
- Context-aware recommendations (Bazi + Weather + Profile)
- DeepSeek for high-quality Bazi reasoning
- Zero-setup persistent memory using SQLite (built into Python)
- Single oracle.db file (no server needed)
- All conversation and venue tracking capabilities
- Image generation via Nano Banana (gemini-2.5-flash-image) with Pollinations.ai free fallback

Graph flow:
  START -> context -> classify -> [recommend | rag | chat | image | sql] -> END
                                    
                                     SQL node for memory queries

Multi-Agent Architecture:
  - Context Agent: Fetches weather and Bazi data
  - Classifier: Routes to appropriate agent (local keyword matching)
  - Recommender Agent: Venue suggestions with Google Places + DeepSeek reasoning
  - RAG Agent: Profile-based Q&A from document chunks
  - Image Agent: Text-to-image generation via Nano Banana + Pollinations fallback
  - Chat Agent: General conversation with DeepSeek
  - SQL Agent: Queries conversation history database

Setup:
    NO INSTALLATION NEEDED - SQLite is built into Python!
    Just run: python strategic_oracle_gemini.py
    
    Database file 'oracle.db' will be created automatically.

Performance:
    Total response time: ~7-10s (optimized from 105s - 90% improvement)
    - Classification: 0.001s (local keyword matching)
    - Keywords: 0.001s (local, profile + context aware)
    - Venue search: ~2s (Google Places API)
    - Reasoning: ~6-9s (DeepSeek  3 venues)
"""

import json
import os
import sys
import urllib.parse
import requests
import datetime

# Force UTF-8 on Windows console so venue names with CJK/special chars don't crash print()
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import math
import time
import warnings
from typing import TypedDict, Annotated, Optional
import operator

# Pre-load google.genai namespace before other packages can corrupt it
import google.genai
import google.genai.types
from google import genai
from google.genai import types as genai_types
from openai import OpenAI
from lunar_python import Lunar

# Import SQLite integration (no external dependencies!)
from oracle_sqlite import OracleDB, make_sql_node, wrap_with_logging

warnings.filterwarnings("ignore")

# =============================================================================
# API CONFIGURATION  loaded from .env file
# =============================================================================
from dotenv import load_dotenv

# Load API keys  tries .env first, then env_config.py as fallback
_script_dir = os.path.dirname(os.path.abspath(__file__))
_env_file   = os.path.join(_script_dir, ".env")
_cfg_file   = os.path.join(_script_dir, "env_config.py")

if os.path.exists(_env_file):
    load_dotenv(_env_file)
    print("[OK] Loaded API keys from .env")
elif os.path.exists(_cfg_file):
    # env_config.py has the same KEY=VALUE format as .env
    load_dotenv(_cfg_file)
    print("[OK] Loaded API keys from env_config.py")
    print("  Tip: rename env_config.py to .env for standard usage")
else:
    print("[WARNING] No .env or env_config.py found  API keys must be set as environment variables")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
PLACES_API_KEY = os.getenv("PLACES_API_KEY", "")
WEATHERAPI_KEY = os.getenv("WEATHERAPI_KEY", "")
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY", "")

if not all([GEMINI_API_KEY, PLACES_API_KEY, WEATHERAPI_KEY, OPENROUTER_KEY]):
    _missing = [k for k, v in {
        "GEMINI_API_KEY": GEMINI_API_KEY,
        "PLACES_API_KEY": PLACES_API_KEY,
        "WEATHERAPI_KEY": WEATHERAPI_KEY,
        "OPENROUTER_KEY": OPENROUTER_KEY,
    }.items() if not v]
    print(f"[WARNING] Missing API keys in .env: {', '.join(_missing)}")
    print("  Copy .env.example to .env and fill in your keys.")


gemini_client  = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL   = "gemini-2.5-flash"

# OpenRouter client for DeepSeek reasoning
openrouter_client = OpenAI(
    api_key=OPENROUTER_KEY,
    base_url="https://openrouter.ai/api/v1"
)

# Model selection: DeepSeek for high-quality Bazi reasoning
REASONING_MODEL = "deepseek/deepseek-chat"  # Used for all LLM reasoning tasks
DEEPSEEK_MODEL = "deepseek/deepseek-chat"   # Alias for backward compatibility

# LangGraph
from langgraph.graph import StateGraph, END

# PDF reading for RAG  pypdf is lightweight and built for text extraction
try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False
    print("[WARNING] pypdf not installed  PDF profile loading disabled. Run: pip install pypdf")

# =============================================================================
# HARDCODED BAZI PROFILE (unchanged)
# =============================================================================
HARDCODED_PROFILE = {
    "name":       "Chen Yingkai",
    "age":        43,
    "occupation": "Digital Banking Professional / AI Student",
    "bazi_chart": (
        "Four Pillars: Year  (Ren Xu), Month  (Xin Hai), "
        "Day  (Bing Chen) - Day Master  Yang Fire, "
        "Hour  (Ding You). "
        "Zodiac: Dog (). Five Element dominant: Fire (). "
        "Season born: between Xiaoxue and Daxue (Water season). "
        "Current Luck Cycle:  Bing Chen (2025-2035) - self-amplification, "
        "identity-driven decisions, living authentically becomes necessary. "
        "2025 annual pillar  (Yi Si): growth and exploration. "
        "2026 annual pillar  (Bing Wu): strong action and visibility - watch overextending. "
        "Chart dynamics: Rob Wealth (), Proper Wealth (), Seven Killings (). "
        "All five elements present. Wood is weaker. Born in Water season so Fire needs support. "
        "Key clash:  (Chen Xu clash) between Day and Year branch - "
        "creates life transitions and identity shifts. "
        "Favourable elements: Wood (growth, learning, teaching) and Fire (expression, purpose). "
        "Day Master  (Yang Fire / Sun Fire): warm, expressive, energising, "
        "wants to guide, illuminate, and impact. Needs purpose and visibility. "
        "Under pressure: intensity becomes anxiety, self-pressure increases."
    ),
    "key_traits": [
        "Deep feeler and deep thinker with high openness (92) and agreeableness (90)",
        "Emotionally sensitive with neuroticism (71) - prone to anxiety under rigid systems",
        "Balanced social energy (extraversion 54) - comfortable alone or with others",
        "Prefers flexibility over rigid systems (conscientiousness 42)",
        "Naturally functions as guide, integrator, and facilitator"
    ],
    "health_notes": (
        "Certified yoga and Pilates instructor. Trained in Thai massage. "
        "Enjoys running and pickleball. Both mentally reflective and physically grounded. "
        "Post-yoga calm is a key restorative state."
    ),
    "goals": (
        "Exploring human-centred work: teaching yoga/Pilates, coaching, facilitation. "
        "Applying AI with purpose in fraud/risk, operations, and social good. "
        "Pursuing Barista FIRE philosophy - sustainable, meaningful work. "
        "Core direction: understand deeply, help others grow, express and guide. "
        "Becoming a life integrating Mind (AI, learning), Body (yoga, somatic), "
        "Heart (teaching, guiding), and Sensory Craft (coffee, food)."
    ),
    "preferences": (
        "Intellectual: AI, philosophy, interconnectedness. "
        "Creative: Kurosawa films, Murakami books, theatre, documentaries. "
        "Music: jazz (trumpet for uplifting, tenor sax for mellow), Autumn Leaves. "
        "Food: vegetarian-leaning, Japanese cuisine, spicy food. "
        "Coffee: certified SCA brewer, treats coffee as mindful aesthetic experience. "
        "Social: meaningful conversations, depth over superficial interaction, "
        "checking on others wellbeing, warmth and presence. "
        "Light: visiting cats with wife, animal-human interaction videos."
    ),
    "summary": (
        "A 43-year-old Singapore-based digital banking professional who took a career break "
        "in 2025 after losing his mother, which deepened his orientation toward meaning, "
        "presence, and intentional living. "
        "He is a sensitive yet driven Sun Fire () individual - certified yoga instructor, "
        "SCA coffee brewer, and AI student - currently in a  luck cycle (2025-2035) "
        "that calls for authentic living and identity-driven decisions. "
        "He is navigating a core tension between staying competent in banking "
        "versus moving toward teaching, coaching, and purposeful AI work."
    )
}

# =============================================================================
# MEMORY & CONFIGURATION
# =============================================================================
# Limit conversation history to last 10 exchanges for context window management
MAX_HISTORY = 6

# Triggers for image generation requests
IMAGE_TRIGGERS = [
    "generate image", "with image", "visualise", "visualize",
    "show image", "illustrate", "generate a picture", "draw me",
    "create an image", "make an image"
]

# Separate triggers that need image + something else (recommend + image)
IMAGE_COMPANION_TRIGGERS = ["show me an image", "with an image", "and show", "show image"]

# Directory for saving generated images
# Always store images relative to this script, not the working directory
IMAGE_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oracle_images")

# Default profile PDF path (place your profile PDF here)
DEFAULT_PROFILE_PDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profile.pdf")

# Style guidance for all LLM responses
STYLE_GUARD = (
    "Do not use any greeting or salutation like My Dear or Dear Friend. "
    "Begin immediately with the substance. Be warm, direct, and specific."
)

# System prompt for Bazi consultation using DeepSeek
BAZI_SYSTEM_PROMPT = (
    "You are an expert Bazi consultant with deep knowledge of classical Chinese metaphysics. "
    "The user is Chen Yingkai, a 43-year-old Sun Fire ( Yang Fire) Day Master. "
    "His chart: Year , Month , Day , Hour . "
    "Current luck cycle  (2025-2035): self-amplification, living authentically. "
    "Wood is weaker in his chart - favourable elements are Wood and Fire. "
    "Key clash:  causes life transitions and identity shifts. "
    "Production cycle: Wood produces Fire produces Earth produces Metal produces Water. "
    "Control cycle: Wood controls Earth, Water controls Fire, Fire controls Metal. "
    "Always explain elemental logic in plain English. Respond in English."
)

# Metaphysics data dictionaries (unchanged)
HEAVENLY_STEMS = {
    "\u7532": "Wood Yang: Expansive, ambitious. Favours bold new starts.",
    "\u4e59": "Wood Yin: Flexible, artistic. Favours creativity and connection.",
    "\u4e19": "Fire Yang: Radiant, social. Favours networking and visibility.",
    "\u4e01": "Fire Yin: Heartfelt, warm. Favours deep conversations and art.",
    "\u620a": "Earth Yang: Stable, grounding. Favours planning and nature.",
    "\u5df1": "Earth Yin: Nurturing, restorative. Favours self-care.",
    "\u5e9a": "Metal Yang: Sharp, disciplined. Favours decisive action.",
    "\u8f9b": "Metal Yin: Refined, aesthetic. Favours beauty and learning.",
    "\u58ec": "Water Yang: Dynamic, adventurous. Favours exploring new places.",
    "\u7678": "Water Yin: Intuitive, introspective. Favours meditation and reflection.",
}

EARTHLY_BRANCHES = {
    "\u5b50": "Rat: sharp intuition, networking.",
    "\u4e11": "Ox: steady, focused work.",
    "\u5bc5": "Tiger: bold, exercise.",
    "\u536f": "Rabbit: gentle, arts.",
    "\u8fb0": "Dragon: dynamic, business.",
    "\u5df3": "Snake: introspective, planning.",
    "\u5348": "Horse: high energy, social outings.",
    "\u672a": "Goat: nurturing, relaxation.",
    "\u7533": "Monkey: clever, learning.",
    "\u9149": "Rooster: precise, refinement.",
    "\u620c": "Dog: loyal, community.",
    "\u4ea5": "Pig: restful, leisure.",
}

TWELVE_OFFICERS = {
    "Establish": "A powerful day to initiate new plans or begin something meaningful.",
    "Remove":    "A day to cleanse and let go of what no longer serves you.",
    "Full":      "An abundant day for celebrations and gatherings.",
    "Balance":   "A neutral day for routine and equilibrium.",
    "Stable":    "A grounded day for firm decisions.",
    "Hold":      "A cautious day for consolidating and reviewing.",
    "Break":     "A volatile day, good for breaking bad habits.",
    "Danger":    "A day for extra caution.",
    "Success":   "An auspicious day for completing goals.",
    "Receive":   "A day for collecting, learning, and absorbing.",
    "Open":      "A highly auspicious day for new beginnings.",
    "Close":     "A day for wrapping up and internal reflection.",
}

CN_OFFICER_MAP = {
    "\u5efa": "Establish", "\u9664": "Remove",  "\u6ee1": "Full",
    "\u5e73": "Balance",   "\u5b9a": "Stable",  "\u57f7": "Hold",
    "\u7834": "Break",     "\u5371": "Danger",  "\u6210": "Success",
    "\u6536": "Receive",   "\u958b": "Open",    "\u9589": "Close",
}

STEM_SECTOR_MAP = {
    "\u7532": "E",  "\u4e59": "SE", "\u4e19": "S",  "\u4e01": "SW",
    "\u620a": "NE", "\u5df1": "NE", "\u5e9a": "W",  "\u8f9b": "NW",
    "\u58ec": "N",  "\u7678": "NE"
}

ELEMENT_VISUAL = {
    "\u7532": "lush ancient forest, towering trees, dappled morning light",
    "\u4e59": "delicate bamboo grove, soft watercolour greens, graceful",
    "\u4e19": "blazing golden sunrise over a city skyline, radiant and energetic",
    "\u4e01": "warm candlelit room, soft amber glow, heartfelt",
    "\u620a": "vast canyon, solid rock formations, golden earth tones",
    "\u5df1": "lush garden courtyard, mossy stones, nurturing",
    "\u5e9a": "gleaming steel mountains, sharp peaks, crisp clear air",
    "\u8f9b": "refined white marble interior, cool silver light, elegant",
    "\u58ec": "dynamic ocean waves, deep blue, flowing and adventurous",
    "\u7678": "misty mountain lake at dawn, deep indigo, calm",
}

# =============================================================================
# PERSONALISED BAZI INTERACTION (for Day Master \u4e19 Yang Fire / Sun Fire)
# =============================================================================
# Five Element cycle: Wood feeds Fire, Fire produces Earth,
# Earth breeds Metal, Metal holds Water, Water controls Fire.
#
# Chen Yingkai's chart: Day Master \u4e19 (Yang Fire), Day Branch \u8fb0 (Dragon/Earth),
# Born Water season, Fire needs support. Favourable: Wood, Fire.
# 2026 annual pillar \u4e19\u5348 (Bing Wu) \u2014 strong Fire year, watch overextension.

# How each day's Heavenly Stem interacts with \u4e19 Yang Fire Day Master
STEM_IMPACT_ON_BING_FIRE = {
    # Wood feeds Fire \u2192 SUPPORTIVE
    "\u7532": {  # \u7532 Jia (Wood Yang)
        "quality": "Growth Energy \u2014 Express & Lead",
        "colour": "#2e7d32",
        "impact": "Expansive, clarifying energy. Good day for your transition: your natural radiance as guide/teacher comes online. Ideas about your next phase flow more easily.",
        "do": "Share something you've been thinking about (to trusted others, not publicly yet). Lead a conversation or class if opportunity shows up. Act on the teaching impulse.",
        "avoid": "Over-explaining your transition. Seeking validation. Assuming your clarity today will last.",
        "training": "Z2 aerobic work. Your body has real fuel\u2014use it for movement that expresses, not just moves.",
        "psyche": "Your openness (92nd percentile) is activated. Notice: what excites you to teach or share? That's the direction signal.",
    },
    "\u4e59": {  # \u4e59 Yi (Wood Yin)
        "quality": "Gentle Growth \u2014 Connect & Create",
        "colour": "#43a047",
        "impact": "Softer than Jia, but deeper. Good for authentic connection with people in your life (not public visibility). Creative work on identity questions flows naturally.",
        "do": "Deep conversation with someone you trust. Write about your transition (for yourself). Teach in an intimate setting (one-on-one). Make something.",
        "avoid": "Forcing momentum. Needing today to 'prove' anything.",
        "training": "Yoga, Pilates, or embodied practice. Not for escape\u2014for noticing what your body knows about your direction.",
        "psyche": "Your agreeableness (90) is well-supported. You can show vulnerability about your uncertainty without losing yourself.",
    },
    # Fire companions \u2192 AMPLIFYING (watch overextension)
    "\u4e19": {  # \u4e19 Bing (Fire Yang) \u2014 same as Day Master
        "quality": "Amplified Radiance \u2014 Use Wisely",
        "colour": "#ef6c00",
        "impact": "Double Fire energy. Your capacity to influence, teach, and express peaks. Also: your intensity and drive can feel urgent/pressuring. 2026 theme: watch for overextending in this cycle.",
        "do": "Lead, teach, or speak about something you care about. Use this visibility for your actual values, not just output. Show up authentically.",
        "avoid": "Burning out by overcommitting. Ego-driven decisions disguised as 'opportunity.' Intensity that masks anxiety (see psyche).",
        "training": "If recovery is good, use it. But notice: are you exercising or pushing? Intensity today can feel like purpose when it's actually anxiety.",
        "psyche": "Neuroticism (71) shows up as urgency. You feel like you need to do/prove something TODAY. Notice this pattern. The real work is pacing, not proving.",
    },
    "\u4e01": {  # \u4e01 Ding (Fire Yin)
        "quality": "Warmly Aligned",
        "colour": "#ff8f00",
        "impact": "Yin Fire alongside your Yang Fire \u2014 warmth without burning. Heart-centred connections thrive.",
        "do": "Deep conversations, mentoring, journaling, coffee rituals, intimate gatherings.",
        "avoid": "Scattering energy across too many people. Focus on depth, not breadth.",
        "training": "Moderate effort works best \u2014 a Z2 run with mindful pacing, or a restorative yoga session.",
        "psyche": "Your facilitator nature activates. Excellent day for coaching or guiding someone.",
    },
    # Fire produces Earth \u2192 OUTPUT / slightly draining
    "\u620a": {  # \u620a Wu (Earth Yang)
        "quality": "Productive but Draining",
        "colour": "#f9a825",
        "impact": "Your Fire feeds Earth \u2014 you'll produce results but it costs energy. Pace yourself.",
        "do": "Planning, organising, completing tasks, administrative work. Productive output day.",
        "avoid": "Overcommitting \u2014 you'll feel capable but fatigue hits later. Build in breaks.",
        "training": "Keep it light \u2014 Z1/Z2 only. Your body is outputting energy, not storing it.",
        "psyche": "Low conscientiousness (42) means structure drains you faster. Use short focused sprints.",
    },
    "\u5df1": {  # \u5df1 Ji (Earth Yin)
        "quality": "Nurturing Output",
        "colour": "#c0ca33",
        "impact": "Your Fire gently warms Earth \u2014 good for giving back, teaching, caring for others.",
        "do": "Self-care, cooking, gardening, helping someone, Thai massage, nurturing activities.",
        "avoid": "Neglecting yourself while caring for others. Put your own oxygen mask on first.",
        "training": "Recovery day. Gentle stretching, foam rolling, or a slow walk. Let the body heal.",
        "psyche": "Your sensitivity is heightened. Honour it \u2014 journal or reflect rather than push through.",
    },
    # Fire controls Metal \u2192 CONFRONTATIONAL / takes effort
    "\u5e9a": {  # \u5e9a Geng (Metal Yang)
        "quality": "Challenging \u2014 Requires Effort",
        "colour": "#e65100",
        "impact": "Your Fire clashes with Metal \u2014 you have power over the situation but it costs energy. Friction is possible.",
        "do": "Tackle difficult conversations, negotiate, cut through indecision. Your Fire can reshape Metal.",
        "avoid": "Unnecessary confrontation. Pick battles wisely \u2014 not everything needs your fire today.",
        "training": "Channel tension into physical effort \u2014 a hard pickleball session or interval training.",
        "psyche": "Seven Killings (\u6bba) energy activated. Stay disciplined but don't let it become aggression.",
    },
    "\u8f9b": {  # \u8f9b Xin (Metal Yin)
        "quality": "Refining",
        "colour": "#ff6f00",
        "impact": "Your Fire refines Yin Metal \u2014 good for precision, editing, and polishing, not for starting new.",
        "do": "Review work, refine ideas, improve existing projects, aesthetic pursuits, coffee cupping.",
        "avoid": "Starting brand new initiatives \u2014 today is for sharpening, not pioneering.",
        "training": "Technique-focused session \u2014 work on running form, breathing drills, or skill practice.",
        "psyche": "Your aesthetic sensitivity peaks. Visit a gallery, listen to jazz, or brew a careful pour-over.",
    },
    # Water controls Fire \u2192 EXTERNAL PRESSURE
    "\u58ec": {  # \u58ec Ren (Water Yang)
        "quality": "Pressure Inward \u2014 Consolidate",
        "colour": "#c62828",
        "impact": "External forces feel constraining. Your Fire is redirected inward rather than dampened \u2014 this is actually good for your transition phase, as it forces reflection over action.",
        "do": "Take inventory of what's working in your current direction (teaching vs. banking). Notice what you're protecting vs. what you're avoiding.",
        "avoid": "Pushing through resistance as if force will clarify things. Dismissing doubts as 'lack of faith.' Making big moves when the environment is pushing back.",
        "training": "Z1-Z2 only. Use this as a signal to rest, not a sign you're failing. Your body is asking for consolidation, not conquest.",
        "psyche": "Anxiety (neuroticism 71) will whisper that pressure = wrongness. Instead, notice: does this pressure feel like external resistance, or is it your own self-doubt? Ground in what you can actually control.",
    },
    "\u7678": {  # \u7678 Gui (Water Yin)
        "quality": "Clarity Deferred \u2014 Listen Inward",
        "colour": "#d32f2f",
        "impact": "Yin Water dissolves certainty. Perfect for your phase: things feel uncertain because you're genuinely in transition, not because you're lost. The haziness is the work itself.",
        "do": "Journal what you're noticing without needing to resolve it. Where are you feeling pulled? What feels alive vs. obligatory? Let patterns emerge, don't force them.",
        "avoid": "Seeking external clarity (advice, tarot, another voice). Treating uncertainty as a problem to solve today. Public announcements about your direction.",
        "training": "Yin yoga or slow movement. Not for 'relaxation'\u2014for noticing. What emerges when you're not producing or proving?",
        "psyche": "Your high agreeableness (90) may pressure you to decide and reassure others. Instead, sit with not-knowing. This is the subconscious work of identity transition.",
    },
}

# How each day's Earthly Branch interacts with \u8fb0 (Chen/Dragon) Day Branch
BRANCH_IMPACT_ON_CHEN = {
    "\u5b50": "Rat (Water) brings intellectual stimulation but pressures your Fire. Stay warm.",
    "\u4e11": "Ox (Earth) is steady and grounding. Good for focused, methodical work.",
    "\u5bc5": "Tiger (Wood) feeds your Fire \u2014 bold energy, good for exercise and new starts.",
    "\u536f": "Rabbit (Wood) nurtures gently. Creative arts and connection flourish.",
    "\u8fb0": "Dragon meets Dragon (self-punishment \u8fb0\u8fb0). Watch for self-criticism and overthinking.",
    "\u5df3": "Snake (Fire) is your companion. Natural flow, warmth, and confidence.",
    "\u5348": "Horse (Fire) ignites strong Fire \u2014 high energy but watch for burnout.",
    "\u672a": "Goat (Earth) brings nurturing calm. Good for rest and reflection.",
    "\u7533": "Monkey (Metal) creates tension with your Dragon. Be diplomatic, avoid arguments.",
    "\u9149": "Rooster (Metal) echoes your Hour Pillar \u9149 \u2014 familiar but watch for sharp words.",
    "\u620c": "Dog CLASHES with Dragon (\u8fb0\u620c\u51b2). Major tension day \u2014 expect disruption, stay centred.",
    "\u4ea5": "Pig (Water) challenges your Fire. Conserve energy, prioritise recovery.",
}

# Officer impact personalised for Yang Fire Day Master
OFFICER_PERSONAL_IMPACT = {
    "Establish": {
        "energy": "Initiating",
        "impact": "Your Sun Fire is given a platform. Start something meaningful \u2014 a project, a conversation, a new habit.",
        "advice": "Lead with purpose today. Your natural radiance draws people in.",
    },
    "Remove": {
        "energy": "Cleansing",
        "impact": "Time to release what dims your Fire \u2014 toxic commitments, stale routines, unspoken truths.",
        "advice": "Declutter physically and emotionally. Write down what you're letting go of.",
    },
    "Full": {
        "energy": "Abundant",
        "impact": "Your warmth attracts abundance. Social connections, celebrations, and gratitude amplify today.",
        "advice": "Share a meal, express appreciation, celebrate small wins. Your Fire warms others.",
    },
    "Balance": {
        "energy": "Neutral",
        "impact": "A day for equilibrium. Your intensity can rest \u2014 no need to shine or push.",
        "advice": "Routine maintenance. Catch up on admin, errands, or steady Z2 training.",
    },
    "Stable": {
        "energy": "Anchoring",
        "impact": "Earth energy grounds your Fire. Make firm decisions you've been deliberating on.",
        "advice": "Commit to something concrete \u2014 sign up, book it, lock it in. Stability feeds progress.",
    },
    "Hold": {
        "energy": "Consolidating",
        "impact": "Pause and review. Your Fire burns more efficiently when you consolidate fuel.",
        "advice": "Review your training data, finances, or goals. Don't start new \u2014 optimise existing.",
    },
    "Break": {
        "energy": "Disruptive",
        "impact": "Volatile energy today \u2014 your Fire can break through barriers or burn bridges. Choose wisely.",
        "advice": "Channel disruption constructively: break a bad habit, end a draining commitment.",
    },
    "Danger": {
        "energy": "Cautious",
        "impact": "External threats to your Fire \u2014 miscommunication, accidents, or emotional triggers.",
        "advice": "Move slowly, double-check important communications, avoid risky decisions. Protect your energy.",
    },
    "Success": {
        "energy": "Completing",
        "impact": "Your Fire burns bright at the finish line. Complete what you started \u2014 submissions, goals, conversations.",
        "advice": "Close open loops today. The satisfaction of completion fuels your next cycle.",
    },
    "Receive": {
        "energy": "Absorbing",
        "impact": "Your Fire draws in fuel. Excellent for learning, receiving feedback, and absorbing new knowledge.",
        "advice": "Read, study, attend a class, or listen deeply. Your openness (92) is fully receptive today.",
    },
    "Open": {
        "energy": "Auspicious",
        "impact": "The most favourable officer for your Sun Fire. Doors open, people respond, opportunities appear.",
        "advice": "Make the ask, send the proposal, start the project. Today has your back.",
    },
    "Close": {
        "energy": "Withdrawing",
        "impact": "Your Fire needs to bank its embers. Turn inward \u2014 reflection, meditation, quiet restoration.",
        "advice": "Avoid new commitments. Journal, meditate, do yin yoga. Tomorrow's Fire will be stronger for it.",
    },
}

# =============================================================================
# LANGGRAPH STATE
# =============================================================================
# State dictionary passed between all agents in the graph
class OracleState(TypedDict):
    query:          str                         # User's input query
    intent:         Optional[str]               # Classified intent (recommend/rag/chat/image/sql)
    meta:           Optional[dict]              # Bazi metadata (day pillar, officer, element)
    weather:        Optional[dict]              # Weather data (temp, condition, UV, etc.)
    history:        Annotated[list, operator.add]  # Conversation history (accumulated)
    response:       Optional[str]              # Generated response
    want_images:    Optional[bool]             # Whether user requested image generation
    garmin_context: Optional[str]             # Formatted Garmin health summary (may be None)
    lab_context:    Optional[str]             # Formatted lab marker summary (may be None)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate compass bearing between two coordinates."""
    dLon = math.radians(lon2 - lon1)
    y = math.sin(dLon) * math.cos(math.radians(lat2))
    x = (math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) -
         math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(dLon))
    return (math.degrees(math.atan2(y, x)) + 360) % 360

def get_compass_sector(bearing):
    """Convert bearing (0-360) to compass direction (N, NE, E, etc.)."""
    sectors = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return sectors[int((bearing + 22.5) % 360 / 45)]

def safe_filename(text, max_len=40):
    """Convert text to safe filename (alphanumeric + underscore/dash only)."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text)[:max_len]

def query_wants_images(query, initial_state_value=False):
    """Check if user query contains image generation triggers, or if already set in state."""
    if initial_state_value:
        return True
    q = query.lower()
    return any(t in q for t in IMAGE_TRIGGERS) or any(t in q for t in IMAGE_COMPANION_TRIGGERS)

def format_history(history, last_n=6):
    """Format conversation history for prompt context (last N messages)."""
    if not history:
        return "No previous conversation."
    return "\n".join(
        f"{'User' if m['role']=='user' else 'Oracle'}: {m['content'][:250]}"
        for m in history[-last_n:]
    )

def build_profile_str():
    """Build comprehensive profile string for LLM prompts."""
    p = HARDCODED_PROFILE
    return (
        f"Name: {p['name']} | Age: {p['age']} | Occupation: {p['occupation']} | "
        f"Bazi Chart: {p['bazi_chart']} | "
        f"Key Traits: {', '.join(p['key_traits'])} | "
        f"Health: {p['health_notes']} | "
        f"Goals: {p['goals']} | "
        f"Preferences: {p['preferences']}"
    )


# =============================================================================
# ORACLE SERVICES
# =============================================================================
class OracleServices:
    def __init__(self, home_lat=1.3521, home_lng=103.8198):
        self.home_lat = home_lat
        self.home_lng = home_lng
        self.profile_source = "dict"   # "dict" or "pdf"
        os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
        self.chunks = self._build_chunks()

    def _text_to_chunks(self, text, step=320, size=400):
        """Split text into overlapping chunks for retrieval."""
        chunks = []
        for i in range(0, len(text), step):
            chunks.append(text[i:i+size])
        return chunks if chunks else [text]

    def _build_chunks_from_profile_dict(self):
        """Build chunks from the hardcoded HARDCODED_PROFILE dict (default fallback)."""
        full_text = "\n\n".join([
            f"LIFE CONTEXT: {HARDCODED_PROFILE['summary']}",
            f"OCCUPATION: {HARDCODED_PROFILE['occupation']}",
            f"BAZI CHART: {HARDCODED_PROFILE['bazi_chart']}",
            f"PERSONALITY TRAITS: {', '.join(HARDCODED_PROFILE['key_traits'])}",
            f"HEALTH AND WELLNESS: {HARDCODED_PROFILE['health_notes']}",
            f"GOALS AND DIRECTION: {HARDCODED_PROFILE['goals']}",
            f"INTERESTS AND PREFERENCES: {HARDCODED_PROFILE['preferences']}",
        ])
        return self._text_to_chunks(full_text)

    def load_profile_from_pdf(self, pdf_path_or_bytes):
        """
        Load a profile PDF and rebuild RAG chunks from it.

        Args:
            pdf_path_or_bytes: Either a file path string OR raw bytes (from Streamlit uploader)

        Returns:
            Tuple (success: bool, message: str, page_count: int)
        """
        if not PYPDF_AVAILABLE:
            return False, "pypdf not installed. Run: pip install pypdf", 0

        try:
            import io
            if isinstance(pdf_path_or_bytes, (bytes, bytearray)):
                reader = PdfReader(io.BytesIO(pdf_path_or_bytes))
            else:
                reader = PdfReader(pdf_path_or_bytes)

            page_count = len(reader.pages)
            if page_count == 0:
                return False, "PDF has no pages.", 0

            # Extract text from all pages
            full_text = ""
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    full_text += f"\n\n[Page {i+1}]\n{page_text.strip()}"

            if not full_text.strip():
                return False, "Could not extract text from PDF. It may be a scanned image PDF.", 0

            # Rebuild chunks from PDF text
            self.chunks = self._text_to_chunks(full_text, step=400, size=600)
            self.profile_source = "pdf"
            print(f"  [OK] Profile loaded from PDF: {page_count} pages, {len(self.chunks)} chunks")
            return True, f"Profile loaded from PDF ({page_count} pages, {len(self.chunks)} chunks).", page_count

        except Exception as e:
            return False, f"Failed to read PDF: {e}", 0

    def _build_chunks(self):
        """
        Build RAG chunks. Priority:
          1. Default profile.pdf if it exists next to the script
          2. Hardcoded HARDCODED_PROFILE dict
        """
        self.profile_source = "dict"

        # Try loading default profile PDF automatically
        if PYPDF_AVAILABLE and os.path.exists(DEFAULT_PROFILE_PDF):
            success, msg, _ = self.load_profile_from_pdf(DEFAULT_PROFILE_PDF)
            if success:
                print(f"  [DOC] Auto-loaded profile from: {DEFAULT_PROFILE_PDF}")
                return self.chunks
            else:
                print(f"  [WARNING] Could not auto-load profile.pdf: {msg}")

        # Fallback to hardcoded profile dict
        print("  [FORM] Using hardcoded profile dict for RAG")
        self.profile_source = "dict"
        return self._build_chunks_from_profile_dict()

    def retrieve_chunks(self, question, top_k=3):
        q_words = set(question.lower().split())
        scored  = []
        for chunk in self.chunks:
            chunk_lower = chunk.lower()
            score = sum(1 for w in q_words if w in chunk_lower and len(w) > 3)
            scored.append((score, chunk))
        scored.sort(reverse=True, key=lambda x: x[0])
        top = [c for s, c in scored[:top_k] if s > 0]
        if not top:
            top = [c for _, c in scored[:2]]
        return top

    def fast_keyword(self, query, meta=None, weather=None):
        """Smart keyword extraction with Bazi, weather, AND profile awareness."""
        # Profile key preferences (extracted from HARDCODED_PROFILE)
        PROFILE_TRAITS = {
            'coffee_expert': True,      # SCA certified brewer
            'vegetarian': True,         # Vegetarian-leaning
            'japanese_cuisine': True,   # Loves Japanese food
            'yoga_instructor': True,    # Certified yoga/Pilates instructor
            'depth_seeker': True,       # Prefers meaningful conversations
            'creative': True,           # Kurosawa films, Murakami, theatre
            'wellness_focused': True,   # Yoga, Thai massage, running
            'craft_appreciator': True,  # Coffee as aesthetic experience
        }
        
        # Common stop words to remove
        stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
                      'your', 'yours', 'yourself', 'he', 'him', 'his', 'himself', 'she', 
                      'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 
                      'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 
                      'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 
                      'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 
                      'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 
                      'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 
                      'against', 'between', 'into', 'through', 'during', 'before', 'after', 
                      'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 
                      'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 
                      'when', 'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 
                      'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 
                      'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 
                      'don', 'should', 'now', 'go', 'get', 'want', 'need', 'looking',
                      'feel', 'feeling', 'felt', 'lonely', 'alone', 'sad', 'happy', 'stressed',
                      'anxious', 'depressed', 'bored', 'tired', 'exhausted', 'confused'}
        
        words = query.lower().split()
        base_keywords = [w.strip('?.,!') for w in words if w.lower() not in stop_words and len(w) > 2]
        
        query_lower = query.lower()
        enhanced_keywords = []
        
        # === EMOTIONAL/SOCIAL STATE QUERIES ===
        if any(word in query_lower for word in ['lonely', 'alone', 'isolated', 'need company', 'need people']):
            if PROFILE_TRAITS['depth_seeker']:
                enhanced_keywords = ["cafe coworking space", "social cafe gathering", "bookstore cafe"]
            else:
                enhanced_keywords = ["popular cafe", "social gathering space", "cafe with events"]
            print(f"   Detected: Social connection need  {enhanced_keywords}")
        
        elif any(word in query_lower for word in ['stressed', 'anxious', 'overwhelmed', 'burnt out', 'exhausted']):
            if PROFILE_TRAITS['wellness_focused']:
                enhanced_keywords = ["spa wellness massage", "yoga studio", "meditation center"]
            else:
                enhanced_keywords = ["spa massage", "quiet park", "relaxation center"]
            print(f"   Detected: Stress relief need  {enhanced_keywords}")
        
        elif any(word in query_lower for word in ['sad', 'down', 'depressed', 'low mood', 'blue']):
            enhanced_keywords = ["bright cafe", "botanical garden", "uplifting restaurant"]
            print(f"   Detected: Mood lift need  {enhanced_keywords}")
        
        elif any(word in query_lower for word in ['bored', 'boring', 'nothing to do', 'restless']):
            if PROFILE_TRAITS['creative']:
                enhanced_keywords = ["art gallery", "bookstore", "theatre cinema"]
            else:
                enhanced_keywords = ["entertainment venue", "activity center", "arcade"]
            print(f"   Detected: Stimulation need  {enhanced_keywords}")
        
        elif any(word in query_lower for word in ['confused', 'lost', 'unclear', 'need clarity', 'stuck']):
            enhanced_keywords = ["quiet cafe", "botanical garden", "library"]
            print(f"   Detected: Clarity need  {enhanced_keywords}")
        
        if enhanced_keywords:
            result_keywords = enhanced_keywords[:3]
            result = ", ".join(result_keywords)
            print(f"   Smart keywords: {result} (Emotional/social state detected, Profile-aware)")
            return result
        
        # === STANDARD KEYWORD EXTRACTION ===
        if base_keywords:
            primary = base_keywords[0]
            enhanced_keywords.append(primary)
        
        if any(word in query_lower for word in ['coffee', 'cafe', 'espresso', 'latte', 'brew']):
            if PROFILE_TRAITS['coffee_expert']:
                if weather and weather.get('is_rainy'):
                    enhanced_keywords.append("specialty coffee cozy indoor")
                elif weather and weather.get('temp_c', 30) > 32:
                    enhanced_keywords.append("specialty coffee air conditioned")
                elif meta and 'Open' in meta.get('officer_name', ''):
                    enhanced_keywords.append("specialty coffee coworking social")
                elif meta and 'Balance' in meta.get('officer_name', ''):
                    enhanced_keywords.append("artisan coffee quiet craft brewing")
                else:
                    enhanced_keywords.append("specialty coffee third wave")
            if any(w in query_lower for w in ['eat', 'food', 'brunch', 'lunch']):
                if PROFILE_TRAITS['vegetarian']:
                    enhanced_keywords.append("vegetarian options")
        
        elif any(word in query_lower for word in ['eat', 'food', 'lunch', 'dinner', 'hungry', 'restaurant']):
            if PROFILE_TRAITS['japanese_cuisine']:
                if weather and weather.get('is_rainy'):
                    enhanced_keywords.append("Japanese restaurant indoor vegetarian")
                elif meta and 'Metal' in meta.get('element_desc', ''):
                    enhanced_keywords.append("Japanese fine dining omakase")
                else:
                    enhanced_keywords.append("Japanese restaurant vegetarian")
            elif PROFILE_TRAITS['vegetarian']:
                enhanced_keywords.append("vegetarian restaurant plant based")
        
        elif any(word in query_lower for word in ['relax', 'chill', 'calm', 'unwind', 'peace', 'destress']):
            if PROFILE_TRAITS['wellness_focused']:
                if weather and weather.get('is_rainy'):
                    enhanced_keywords.append("wellness spa massage indoor")
                elif weather and weather.get('is_outdoor_safe'):
                    enhanced_keywords.append("nature park mindful walking")
                else:
                    enhanced_keywords.append("wellness spa yoga retreat")
        
        elif any(word in query_lower for word in ['yoga', 'exercise', 'workout', 'fitness', 'stretch', 'pilates']):
            if PROFILE_TRAITS['yoga_instructor']:
                if meta and '' in meta.get('day_pillar', ''):
                    enhanced_keywords.append("hot yoga vinyasa dynamic")
                elif meta and '' in meta.get('day_pillar', ''):
                    enhanced_keywords.append("yin yoga restorative flow")
                else:
                    enhanced_keywords.append("yoga studio mindful hatha")
        
        elif any(word in query_lower for word in ['movie', 'film', 'theatre', 'art', 'gallery', 'museum', 'culture']):
            if PROFILE_TRAITS['creative']:
                enhanced_keywords.append("art house cinema independent film")
            if PROFILE_TRAITS['depth_seeker']:
                enhanced_keywords.append("documentary thought provoking")
        
        elif any(word in query_lower for word in ['work', 'study', 'focus', 'productive', 'concentrate']):
            if meta and meta.get('officer_name') in ['Danger', 'Break']:
                enhanced_keywords.append("quiet library safe peaceful")
            elif PROFILE_TRAITS['depth_seeker']:
                enhanced_keywords.append("coworking meaningful conversation")
            else:
                enhanced_keywords.append("coworking cafe quiet focus")
        
        elif any(word in query_lower for word in ['nature', 'park', 'outdoor', 'green', 'walk', 'hike', 'garden']):
            if PROFILE_TRAITS['wellness_focused']:
                if weather and weather.get('is_rainy'):
                    enhanced_keywords.append("indoor garden conservatory greenhouse")
                elif weather and weather.get('uv_index', 0) > 8:
                    enhanced_keywords.append("shaded park covered walkway forest")
                else:
                    enhanced_keywords.append("botanic gardens mindful walking park")
        
        elif any(word in query_lower for word in ['meet', 'social', 'friends', 'gathering', 'hangout']):
            if PROFILE_TRAITS['depth_seeker']:
                enhanced_keywords.append("intimate venue conversation friendly")
            else:
                enhanced_keywords.append("social venue gathering space")
        
        # === ACTIVITY + WEATHER QUERIES ===
        elif any(word in query_lower for word in ['activity', 'activities', 'what can i do',
                                                   'suits the weather', 'suits today', 'what to do today']):
            if weather and weather.get('is_rainy'):
                # Rainy  indoor spaces suited to profile: caf, museum, yoga, bookstore
                if PROFILE_TRAITS['coffee_expert']:
                    enhanced_keywords = ["specialty coffee caf indoor cosy", "museum art gallery singapore"]
                else:
                    enhanced_keywords = ["indoor wellness centre singapore", "museum art gallery singapore"]
            elif weather and weather.get('uv_index', 0) > 8:
                # High UV  shaded or indoor
                enhanced_keywords = ["indoor sport fitness singapore", "covered park shaded trail"]
            elif weather and weather.get('is_outdoor_safe'):
                # Good weather  outdoor based on profile
                if PROFILE_TRAITS['yoga_instructor']:
                    enhanced_keywords = ["outdoor yoga park singapore", "botanic gardens nature trail"]
                else:
                    enhanced_keywords = ["outdoor activity park singapore", "nature reserve trail"]
            else:
                enhanced_keywords = ["indoor activity singapore", "wellness centre caf"]
            print(f"   Activity+weather  {enhanced_keywords}")

        else:
            if not base_keywords:
                if PROFILE_TRAITS['coffee_expert']:
                    enhanced_keywords = ["specialty coffee", "wellness"]
                else:
                    enhanced_keywords = ["cafe", "wellness centre"]
            elif len(base_keywords) == 1:
                enhanced_keywords.append("singapore recommended")
        
        result_keywords = enhanced_keywords[:3] if enhanced_keywords else base_keywords[:2]
        result = ", ".join(result_keywords) if result_keywords else "specialty coffee, wellness"
        
        context_info = ""
        if meta or weather:
            context_info = " ("
            if meta:
                context_info += f"{meta.get('officer_name', 'Balance')} Officer"
            if weather:
                if weather.get('is_rainy'):
                    context_info += ", Rainy"
                elif weather.get('temp_c', 30) > 32:
                    context_info += f", Hot {weather['temp_c']}C"
                elif weather.get('uv_index', 0) > 8:
                    context_info += f", High UV"
            context_info += ", Profile-aware"
            context_info += ")"
        
        print(f"   Smart keywords: {result}{context_info}")
        return result

    def gemini(self, prompt):
        for attempt in range(3):
            try:
                r = gemini_client.models.generate_content(
                    model=GEMINI_MODEL, contents=prompt
                )
                return r.text.strip()
            except Exception as e:
                if "429" in str(e):
                    time.sleep(16 * (attempt + 1))
                else:
                    print(f"Gemini error: {e}")
                    return ""
        return ""

    def gemini_vision(self, prompt: str, image_bytes: bytes) -> str:
        """Send a text prompt + image bytes to Gemini for vision extraction."""
        try:
            import base64
            image_b64 = base64.b64encode(image_bytes).decode()
            contents = [
                genai_types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                prompt,
            ]
            r = gemini_client.models.generate_content(
                model=GEMINI_MODEL, contents=contents
            )
            return r.text.strip()
        except Exception as e:
            print(f"  [gemini_vision] error: {e}")
            return ""

    def deepseek(self, prompt, system=BAZI_SYSTEM_PROMPT):
        """DeepSeek for complex reasoning."""
        for attempt in range(3):
            try:
                r = openrouter_client.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": prompt}
                    ]
                )
                return r.choices[0].message.content.strip()
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    time.sleep(16 * (attempt + 1))
                else:
                    print(f"DeepSeek error: {e}")
                    return ""
        return ""

    def minimax(self, prompt, system="Chen YingkaiYou provide warm, insightful Bazi guidance in mixed Chinese-English.", max_tokens=500):
        """
        DeepSeek reasoning for quality Bazi insights.
        Method name kept as 'minimax' for backward compatibility.
        """
        for attempt in range(3):
            try:
                _t0 = time.time()
                r = openrouter_client.chat.completions.create(
                    model=REASONING_MODEL,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=max_tokens,
                )
                _elapsed = time.time() - _t0
                _usage = getattr(r, 'usage', None)
                _tokens = _usage.completion_tokens if _usage else '?'
                _msg = f"  [minimax] DeepSeek via OpenRouter: {_elapsed:.2f}s ({_tokens} tokens out)"
                print(_msg, flush=True)
                with open("debug_recommend.log", "a") as _f:
                    _f.write(_msg + "\n")
                return r.choices[0].message.content.strip()
            except Exception as e:
                print(f"Reasoning error (attempt {attempt+1}): {e}")
                if "429" in str(e) or "rate" in str(e).lower():
                    time.sleep(8 * (attempt + 1))
                else:
                    import traceback
                    traceback.print_exc()
                    if attempt == 2:
                        return ""
        return ""

    def interpret_query_with_qwen(self, query: str, meta: dict, weather: dict) -> dict:
        """
        TIER 1 INTERPRETATION LAYER: Uses Qwen 2.5 72B to interpret user intent
        before passing to DeepSeek for Bazi reasoning.

        This improves recommendation quality by:
        1. Clarifying user intent (relax, dine, exercise, explore, etc.)
        2. Extracting refined keywords/search terms
        3. Identifying optimal venue type for the intent
        4. Considering current health/weather context

        Returns structured dict with:
        - intent: user's underlying goal
        - venue_type: restaurant, park, cafe, gym, outdoor, etc.
        - refined_keywords: better search terms
        - focus_area: what aspect to emphasize in recommendation
        """

        qwen_system = """You are a smart query interpreter for a personalized venue recommendation system.
Your job is to understand what the user really wants and provide structured guidance for a better search.

Analyze the user's query in the context of their current health, weather, and Bazi energy.
Be concise and specific - avoid generic advice.
Respond ONLY in the exact JSON format requested, with no extra text."""

        qwen_prompt = f"""Given this context:
- User query: {query}
- Today's Bazi energy: {meta.get('element_desc', 'balanced')}
- Today's pillar: {meta.get('day_pillar', 'unknown')}
- Current weather: {weather.get('condition', 'unknown')}, {weather.get('temp_c', 25)}C
- Is it raining? {weather.get('is_rainy', False)}

Interpret the user's intent and provide search guidance:

Return ONLY a JSON object (no markdown, no code blocks) with:
{{
  "intent": "string describing what they want (relax, dine, exercise, explore, recover, work, socialize)",
  "venue_type": "most suitable venue category (restaurant, park, cafe, gym, mall, beach, museum, nature, indoor, outdoor)",
  "refined_keywords": "comma-separated specific search terms (e.g., 'quiet cafe, dim lighting' or 'fine dining, seafood' or 'hiking trail, moderate difficulty')",
  "emphasis": "what to emphasize in the final recommendation (Bazi balance, health recovery, social experience, nature connection, stress relief)",
  "must_avoid": "any constraints (e.g., 'avoid crowds' or 'needs parking' or 'must have wifi')"
}}"""

        for attempt in range(2):
            try:
                r = openrouter_client.chat.completions.create(
                    model="openrouter/qwen/qwen-2.5-72b-instruct",
                    messages=[
                        {"role": "system", "content": qwen_system},
                        {"role": "user", "content": qwen_prompt}
                    ],
                    temperature=0.5
                )
                response_text = r.choices[0].message.content.strip()

                # Parse JSON response (json is already imported at module level)
                result = json.loads(response_text)
                print(f"  [qwen-tier1] Intent: {result.get('intent', 'unknown')}, Type: {result.get('venue_type', 'unknown')}")
                return result

            except json.JSONDecodeError as je:
                print(f"  [qwen-tier1] JSON parse error (attempt {attempt+1}): {je}")
                print(f"    Raw response: {response_text[:200] if 'response_text' in locals() else 'no response'}")
                if attempt == 1:
                    # Fallback: return basic structure
                    return {
                        "intent": "venue recommendation",
                        "venue_type": "general",
                        "refined_keywords": query,
                        "emphasis": "alignment with energy",
                        "must_avoid": ""
                    }
            except Exception as e:
                print(f"  [qwen-tier1] Error (attempt {attempt+1}): {e}")
                if "429" in str(e) or "rate" in str(e).lower():
                    time.sleep(5 * (attempt + 1))
                else:
                    if attempt == 1:
                        # Fallback structure
                        return {
                            "intent": "venue recommendation",
                            "venue_type": "general",
                            "refined_keywords": query,
                            "emphasis": "alignment with energy",
                            "must_avoid": ""
                        }

        # Final fallback
        return {
            "intent": "venue recommendation",
            "venue_type": "general",
            "refined_keywords": query,
            "emphasis": "alignment with energy",
            "must_avoid": ""
        }

    def interpret_query_with_gemini(self, query: str, meta: dict, weather: dict, history: list = None) -> dict:
        """
        TIER 1 INTERPRETATION LAYER: Uses Gemini 1.5 Flash to interpret user intent
        before passing to DeepSeek for Bazi reasoning.

        This improves recommendation quality by:
        1. Clarifying user intent (relax, dine, exercise, explore, etc.)
        2. Extracting refined keywords/search terms
        3. Identifying optimal venue type for the intent
        4. Considering current health/weather context

        Returns structured dict with:
        - intent: user's underlying goal
        - venue_type: restaurant, park, cafe, gym, outdoor, etc.
        - refined_keywords: better search terms
        """

        with open("debug_recommend.log", "a") as f:
            f.write(f"[GEMINI-START] Calling Gemini with query: {query[:80]}\n")

        # Extract last recommendation context so follow-ups like "another option" stay on topic
        last_rec_context = ""
        if history:
            recent = [m for m in history[-6:] if m.get("role") == "assistant"]
            if recent:
                last_content = recent[-1].get("content", "")
                if any(w in last_content.lower() for w in ["restaurant", "cafe", "park", "museum", "gym", "bar", "hawker"]):
                    last_rec_context = f"\nPrevious recommendation context: {last_content[:300]}\nIf the user asks for 'another option', maintain the same venue_type as the previous recommendation."

        gemini_prompt = f"""You are a smart query interpreter. Generate keywords that Google Places API can search for.

User query: {query}
Bazi energy: {meta.get('element_desc', 'balanced')}
Weather: {weather.get('condition', 'unknown')}, {weather.get('temp_c', 25)}C{last_rec_context}

Return ONLY JSON (no markdown, no explanation):
{{
  "intent": "relax, dine, exercise, explore, recover, work, or socialize",
  "venue_type": "restaurant, park, cafe, gym, mall, beach, museum, nature, garden, trail, or outdoor",
  "refined_keywords": "1-4 short search terms Google Places understands. Use ONLY the place category noun — never add adjectives. GOOD: 'Japanese restaurant, seafood restaurant, rooftop bar'. BAD: 'trendy restaurant, vibrant dining, new fusion place'. Think: what would you type into Google Maps?",
  "emphasis": "what matters most",
  "must_avoid": "things to avoid or empty string"
}}"""

        for attempt in range(2):
            try:
                # Use the gemini_client that's already configured at module level
                r = openrouter_client.chat.completions.create(
                    model="openai/gpt-4o-mini",
                    messages=[{"role": "user", "content": gemini_prompt}],
                    temperature=0.3,
                    response_format={"type": "json_object"},
                )
                response_text = r.choices[0].message.content.strip()

                # DEBUG: Log raw response
                with open("debug_recommend.log", "a") as f:
                    f.write(f"[GPT4O-MINI-RAW] Response text: {response_text[:300]}\n")

                # Parse JSON response
                result = json.loads(response_text)
                print(f"[OK] INTENT-TIER1: GPT-4o Mini")
                print(f"  Intent: {result.get('intent', 'unknown')}")
                print(f"  Venue Type: {result.get('venue_type', 'unknown')}")
                print(f"  Keywords: {result.get('refined_keywords', 'unknown')}")
                return result

            except json.JSONDecodeError as je:
                print(f"  [gpt4o-mini] JSON parse error (attempt {attempt+1}): {je}")
                with open("debug_recommend.log", "a") as f:
                    f.write(f"[GPT4O-MINI-ERROR] JSON parse failed: {response_text[:300]}\n")
                if attempt == 0:
                    continue
                # Fallback after 2 attempts
                return {
                    "intent": "venue recommendation",
                    "venue_type": "general",
                    "refined_keywords": query,
                    "emphasis": "alignment with energy",
                    "must_avoid": ""
                }
            except Exception as e:
                print(f"  [gemini-tier1] Error (attempt {attempt+1}): {e}")
                with open("debug_recommend.log", "a") as f:
                    f.write(f"[GEMINI-EXCEPTION] Attempt {attempt+1}: {type(e).__name__}: {str(e)[:200]}\n")
                if attempt == 0:
                    time.sleep(2)
                    continue

        # Final fallback
        return {
            "intent": "venue recommendation",
            "venue_type": "general",
            "refined_keywords": query,
            "emphasis": "alignment with energy",
            "must_avoid": ""
        }

    def get_weather(self):
        try:
            url  = (f"https://api.weatherapi.com/v1/current.json"
                    f"?key={WEATHERAPI_KEY}&q={self.home_lat},{self.home_lng}&aqi=no")
            data = requests.get(url, timeout=5).json()
            c    = data["current"]
            cond = c["condition"]["text"]
            w = {
                "condition":       cond,
                "temp_c":          round(c["temp_c"]),
                "feels_like":      round(c["feelslike_c"]),
                "humidity":        c["humidity"],
                "uv_index":        c.get("uv", 0),
                "wind_kph":        c.get("wind_kph", 0),
                "is_rainy":        c.get("precip_mm", 0) > 0 or "rain" in cond.lower(),
                "is_outdoor_safe": c.get("precip_mm", 0) == 0 and c.get("uv", 0) <= 8,
            }
            print(f"Weather: {w['condition']}, {w['temp_c']}C, UV {w['uv_index']}")
            return w
        except Exception as e:
            print(f"Weather failed: {e}")
            return {"condition": "partly cloudy", "temp_c": 30, "feels_like": 33,
                    "humidity": 80, "uv_index": 6, "wind_kph": 10,
                    "is_rainy": False, "is_outdoor_safe": True}

    def get_today_meta(self):
        lunar      = Lunar.fromDate(datetime.datetime.now())
        pillar     = lunar.getDayInGanZhi()
        stem       = pillar[0]
        branch     = pillar[1] if len(pillar) > 1 else ""
        officer_cn = lunar.getZhiXing()
        officer    = CN_OFFICER_MAP.get(officer_cn, "Balance")

        # Personalised impact for Day Master 丙 (Yang Fire)
        stem_impact    = STEM_IMPACT_ON_BING_FIRE.get(stem, {})
        branch_impact  = BRANCH_IMPACT_ON_CHEN.get(branch, "")
        officer_impact = OFFICER_PERSONAL_IMPACT.get(officer, {})

        return {
            "date":              str(datetime.date.today()),
            "day_pillar":        pillar,
            "stem":              stem,
            "element_desc":      HEAVENLY_STEMS.get(stem, "Balanced energy."),
            "branch_energy":     EARTHLY_BRANCHES.get(branch, ""),
            "officer_name":      officer,
            "officer_meaning":   TWELVE_OFFICERS.get(officer, "A balanced day."),
            "auspicious_sector": STEM_SECTOR_MAP.get(stem, "SE"),
            "visual":            ELEMENT_VISUAL.get(stem, "serene balanced landscape"),
            # Personalised fields
            "day_quality":       stem_impact.get("quality", "Balanced"),
            "quality_colour":    stem_impact.get("colour", "#c9a84c"),
            "personal_impact":   stem_impact.get("impact", ""),
            "do_today":          stem_impact.get("do", ""),
            "avoid_today":       stem_impact.get("avoid", ""),
            "training_advice":   stem_impact.get("training", ""),
            "psyche_note":       stem_impact.get("psyche", ""),
            "branch_note":       branch_impact,
            "officer_energy":    officer_impact.get("energy", ""),
            "officer_personal":  officer_impact.get("impact", ""),
            "officer_advice":    officer_impact.get("advice", ""),
        }

    def search_places(self, keywords, venue_type=None):
        """
        Search Google Places with optional type filtering from Qwen interpretation.
        Returns venues ranked by relevance, filtered by venue type if specified.

        venue_type: from Qwen interpretation (restaurant, park, cafe, gym, mall, etc.)
        """
        all_places = []
        seen       = set()

        # Map Qwen venue_type to Google Places includedTypes
        type_mapping = {
            "restaurant": ["restaurant", "fast_food_restaurant", "cafe"],
            "park": ["park", "tourist_attraction", "nature"],
            "cafe": ["cafe", "bakery", "coffee_shop"],
            "gym": ["gym", "fitness_center", "sports_complex"],
            "mall": ["shopping_mall", "shopping_center"],
            "beach": ["beach", "tourist_attraction"],
            "museum": ["museum", "art_gallery"],
            "nature": ["park", "tourist_attraction", "nature"],
            "indoor": ["museum", "art_gallery", "library", "shopping_mall", "cafe", "restaurant"],
            "outdoor": ["park", "tourist_attraction", "nature", "sports_complex", "beach"],
            "general": None  # No type filter
        }

        included_types = type_mapping.get(venue_type, None)
        print(f"[SEARCH] venue_type='{venue_type}'")
        if venue_type and included_types:
            print(f"  => Filtering by includedTypes: {included_types}")
        else:
            print(f"  => NO TYPE FILTER (general search)")

        url = "https://places.googleapis.com/v1/places:searchText"
        headers = {
            "Content-Type":     "application/json",
            "X-Goog-Api-Key":   PLACES_API_KEY,
            "X-Goog-FieldMask": (
                "places.displayName,places.generativeSummary,"
                "places.location,places.formattedAddress,places.types,"
                "places.rating,places.userRatingCount"
            )
        }

        def _fetch_keyword(kw):
            body = {
                "textQuery":      f"{kw} in Singapore",
                "maxResultCount": 3,
                "locationBias": {
                    "circle": {
                        "center": {"latitude": self.home_lat, "longitude": self.home_lng},
                        "radius": 20000.0
                    }
                }
            }
            try:
                resp = requests.post(url, headers=headers, json=body).json()
                print(f"  [API] Keyword '{kw}': Google returned {len(resp.get('places', []))} results")
                return resp.get("places", [])
            except Exception as e:
                print(f"Places search failed for '{kw}': {e}")
                return []

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=3) as pool:
            results_per_kw = list(pool.map(_fetch_keyword, keywords[:3]))

        unwanted_types = {"store", "shopping_mall", "clothing_store", "jewelry_store",
                          "electronics_store", "event_space", "function_establishment",
                          "event_venue", "banquet_hall", "conference_center",
                          "wedding_venue", "catering_service"}
        good_types = {"cafe", "restaurant", "bar", "park", "gym", "spa", "museum",
                      "art_gallery", "library", "coworking_space", "tourist_attraction",
                      "nature", "garden"}

        for places_list in results_per_kw:
            for p in places_list:
                name = p.get("displayName", {}).get("text", "")
                if not name or name in seen:
                    continue
                seen.add(name)

                place_types = p.get("types", [])
                print(f"    [OK] '{name}' (types: {place_types[:3]})")

                score = 0
                rating = p.get("rating", 0)
                if rating >= 4.0:
                    score += 3
                elif rating >= 3.5:
                    score += 1

                review_count = p.get("userRatingCount", 0)
                if review_count >= 100:
                    score += 2
                elif review_count >= 50:
                    score += 1

                summary = p.get("generativeSummary", {}).get("overview", {}).get("text", "")
                if summary and len(summary) > 50:
                    score += 2

                if any(t in place_types for t in unwanted_types):
                    score -= 10
                if any(t in place_types for t in good_types):
                    score += 2

                if "ChillOut" in name or "Chill Out" in name:
                    print(f"  [DEBUG] {name}: types={place_types}, score={score}")

                p["_relevance_score"] = score

                loc = p.get("location", {})
                bearing = calculate_bearing(
                    self.home_lat, self.home_lng,
                    loc.get("latitude", self.home_lat),
                    loc.get("longitude", self.home_lng)
                )
                p["_sector"] = get_compass_sector(bearing)
                all_places.append(p)
        
        all_places.sort(key=lambda x: x.get("_relevance_score", 0), reverse=True)
        return all_places

    # =========================================================================
    # IMAGE GENERATION  Nano Banana primary, Pollinations as free fallback
    # =========================================================================
    def generate_image(self, prompt, filename):
        """
        Generate image with a 2-tier fallback chain:
          1. Nano Banana (gemini-2.5-flash-image)  $0.039/image, GA (PRIMARY)
          2. Pollinations.ai          free, no key, fallback if Gemini quota fails
        """
        # Ensure output directory exists
        abs_dir = IMAGE_OUTPUT_DIR  # Already absolute, set at module level
        os.makedirs(abs_dir, exist_ok=True)
        print(f"  [image] Output directory: {abs_dir}")

        # --- Tier 1: Gemini 2.5 Flash Image aka Nano Banana ($0.039/image, GA) ---
        try:
            print(f"  [image] Generating via Gemini 2.5 Flash Image (Nano Banana): {prompt[:80]}...")
            resp1 = gemini_client.models.generate_content(
                model="gemini-2.5-flash-image",
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=genai_types.ImageConfig(
                        aspect_ratio="16:9",
                    ),
                ),
            )
            # Debug: show what parts came back
            parts = resp1.candidates[0].content.parts
            print(f"  [image] Nano Banana returned {len(parts)} part(s)")
            for i, part in enumerate(parts):
                if part.inline_data is not None:
                    mime = part.inline_data.mime_type
                    size = len(part.inline_data.data)
                    print(f"  [image] Part {i}: inline_data mime={mime}, size={size} bytes")
                    filepath = os.path.join(abs_dir, f"{filename}.png")
                    with open(filepath, "wb") as f:
                        f.write(part.inline_data.data)
                    print(f"  [OK] Nano Banana image saved: {filepath}")
                    return filepath
                else:
                    text_preview = (part.text[:80] if hasattr(part, "text") and part.text else "no text")
                    print(f"  [image] Part {i}: text part  '{text_preview}'")
            print("  [WARNING] Nano Banana returned no image part  falling back.")
        except Exception as e:
            print(f"  [WARNING] Nano Banana (gemini-2.5-flash-image) failed: {e}")

        # --- Tier 2: Pollinations.ai (free fallback) ---
        try:
            print(f"  [image] Falling back to Pollinations.ai...")
            encoded = urllib.parse.quote(prompt)
            url = (f"https://image.pollinations.ai/prompt/{encoded}"
                   f"?width=1280&height=720&nologo=true&enhance=true")
            resp2 = requests.get(url, timeout=60)
            resp2.raise_for_status()
            content_type = resp2.headers.get("Content-Type", "unknown")
            size = len(resp2.content)
            print(f"  [image] Pollinations response: content-type={content_type}, size={size} bytes")
            if "image" not in content_type:
                print(f"  [WARNING] Pollinations returned non-image content ({content_type}), skipping save.")
            else:
                filepath = os.path.join(abs_dir, f"{filename}_pollinations.png")
                with open(filepath, "wb") as f:
                    f.write(resp2.content)
                print(f"  [OK] Pollinations image saved: {filepath}")
                return filepath
        except Exception as e:
            print(f"  [WARNING] Pollinations failed: {e}")

        print("  [FAIL] All image generation methods exhausted.")
        return None


# =============================================================================
# LANGGRAPH NODES
# =============================================================================
# =============================================================================
# LLM FALLBACK CLASSIFIER
# =============================================================================
def _llm_classify(query: str, history: list, svc) -> str:
    """
    LLM-based intent classifier using Gemini Flash.
    Called only when keyword matching produces no confident result.

    Returns one of: recommend | rag | chat | image | sql

    Uses a tight structured prompt so the model returns a single word,
    keeping latency low (~0.5-1s) and cost negligible.
    """
    # Build minimal history context (last 2 exchanges only)
    ctx_lines = []
    recent = [m for m in history[-4:] if m.get("role") in ("user", "assistant")]
    for m in recent:
        role = "User" if m["role"] == "user" else "Oracle"
        ctx_lines.append(f"{role}: {m['content'][:120]}")
    context_str = "\n".join(ctx_lines) if ctx_lines else "No prior context."

    prompt = f"""You are a query intent classifier for a Bazi oracle chatbot.
Classify the user query into exactly ONE of these intents:

  recommend  user wants a venue, activity, or place suggestion
  rag        user asks about their own profile, Bazi chart, traits, or goals
  chat       follow-up question, general conversation, or asking about a previously recommended place
  image      user explicitly wants an image generated
  sql        user asks about conversation history, past venues, or memory

Recent conversation:
{context_str}

User query: "{query}"

Reply with ONE word only. No explanation. Choose from: recommend, rag, chat, image, sql"""

    try:
        result = svc.gemini(prompt)
        cleaned = result.strip().lower().split()[0] if result else "chat"
        valid = {"recommend", "rag", "chat", "image", "sql"}
        return cleaned if cleaned in valid else "chat"
    except Exception as e:
        print(f"  [classify] LLM fallback failed: {e}  defaulting to chat")
        return "chat"


def detect_time_period(query: str) -> dict:
    """
    Detect the time period the user is asking about.
    Returns dict with keys:
      period      : "today" | "yesterday" | "2days" | "this_week" | "last_week" | "month"
      start_date  : ISO date string (inclusive lower bound for activity queries)
      end_date    : ISO date string (inclusive upper bound)
      days        : legacy count kept for stats queries that still use it

    NOTE: Week/month references are checked BEFORE single-day references so that
    a query like "how did I fare this week" is not hijacked by the word "today"
    appearing in the same sentence.
    """
    import datetime as _dt
    today = _dt.date.today()
    query_lower = query.lower()

    def _iso(d): return d.isoformat()

    # ── Week / month references checked FIRST ──────────────────────────────
    # "last week" = previous calendar week Monday–Sunday
    if any(w in query_lower for w in ["last week", "previous week", "prior week"]):
        days_since_monday = today.weekday()          # Mon=0 … Sun=6
        this_monday = today - _dt.timedelta(days=days_since_monday)
        last_mon = this_monday - _dt.timedelta(days=7)
        last_sun = this_monday - _dt.timedelta(days=1)
        return {"period": "last_week", "days": 7, "offset_days": 0,
                "start_date": _iso(last_mon), "end_date": _iso(last_sun)}

    # "this week" = current calendar week Monday–today
    elif any(w in query_lower for w in ["this week", "past week", "week's", "fared",
                                         "weekly recap", "week recap", "week summary", "weekly review"]):
        days_since_monday = today.weekday()
        this_monday = today - _dt.timedelta(days=days_since_monday)
        days_count = days_since_monday + 1   # at least 1
        return {"period": "this_week", "days": days_count, "offset_days": 0,
                "start_date": _iso(this_monday), "end_date": _iso(today)}

    # Month references
    elif any(w in query_lower for w in ["last month", "this month", "past month", "monthly"]):
        return {"period": "month", "days": 30, "offset_days": 0,
                "start_date": _iso(today - _dt.timedelta(days=30)), "end_date": _iso(today)}

    # ── Single-day references ───────────────────────────────────────────────
    elif any(w in query_lower for w in ["yesterday", "yesterday's", "yest"]):
        yest = today - _dt.timedelta(days=1)
        return {"period": "yesterday", "days": 1, "offset_days": 1,
                "start_date": _iso(yest), "end_date": _iso(yest)}

    elif any(w in query_lower for w in ["today", "today's", "this morning", "this evening"]):
        return {"period": "today", "days": 1, "offset_days": 0,
                "start_date": _iso(today), "end_date": _iso(today)}

    elif any(w in query_lower for w in ["two days", "2 days", "couple days"]):
        return {"period": "2days", "days": 2, "offset_days": 0,
                "start_date": _iso(today - _dt.timedelta(days=1)), "end_date": _iso(today)}

    # Default: today + yesterday for context (covers most "how am I doing" queries)
    return {"period": "today", "days": 2, "offset_days": 0,
            "start_date": _iso(today - _dt.timedelta(days=1)), "end_date": _iso(today)}


def _format_lab_context(lab_summary: dict) -> str:
    """Format lab marker summary as a compact string for LLM prompts."""
    lines = [" Lab Health Markers "]
    abnormal = lab_summary.get("abnormal", [])
    if abnormal:
        lines.append("Flagged markers (abnormal / suboptimal):")
        for m in abnormal:
            ref = ""
            if m.get("ref_high"):
                ref = f" (ref < {m['ref_high']} {m['unit']})"
            elif m.get("ref_low"):
                ref = f" (ref > {m['ref_low']} {m['unit']})"
            lines.append(
                f"   {m['marker_name']} ({m['marker_code']}): "
                f"{m['value']} {m['unit']}{ref} [{m['flag']}]  {m['collected_at']}"
            )
    else:
        lines.append("All measured markers within optimal or normal range.")
    return "\n".join(lines)


# Keywords that signal the user's query is health / medical in nature.
# Lab context is only injected when one of these is present.
_HEALTH_QUERY_KEYWORDS = [
    # Lab marker names / codes
    "ldl", "hdl", "cholesterol", "triglyceride", "lipoprotein", "lp(a)", "lpa",
    "apolipoprotein", "apo b", "homocysteine", "hba1c", "glucose", "blood sugar",
    "crp", "inflammation", "thyroid", "tsh", "vitamin d", "vitamin b", "folate",
    "ferritin", "egfr", "kidney", "ctdna", "cancer marker", "blood work",
    "blood test", "lab result", "lab marker", "lab report",
    # Health conditions / outcomes
    "cardiovascular", "heart health", "heart disease", "atherosclerosis",
    "cardiometabolic", "metabolic", "longevity marker", "biological age",
    "risk factor", "supplement", "medication", "rosuvastatin", "statin",
    # Diet / nutrition in health context
    "diet", "nutrition", "omega", "vegetable", "leafy green", "red meat",
    "alcohol", "caffeine", "methylat", "b complex", "folate supplement",
    # Fitness / activity / recovery
    "fitness", "fitness readiness", "training readiness", "training status",
    "heart rate", "hrv", "heart rate variability", "resting heart rate",
    "vo2", "vo2 max", "aerobic capacity", "cardio fitness", "aerobic fitness",
    "zone 2", "training load", "recovery", "overtraining", "body battery",
    "endurance", "max heart rate", "performance", "VO2 max", "how have i been moving",
    "how am i doing", "my workouts", "recent activities", "how active",
    "activity summary", "movement", "activity", "exercise", "workout",
    "breathing rate", "respiration", "sleep quality", "sleep",
    # Weekly / period reviews
    "how did i fare", "how did i do", "how have i done", "how i fared",
    "this week", "last week", "past week", "weekly recap", "week recap",
    "week summary", "week's activities", "weekly review", "this month",
    "how was my week", "run", "running", "went for a",
    # Body composition / metrics
    "body composition", "body fat", "muscle percent", "weight", "body weight",
    "blood pressure", "systolic", "diastolic",
    # Symptoms / energy levels with medical nuance
    "fatigue", "low energy", "brain fog", "cognitive", "memory",
    # Explicit health queries
    "my health", "my results", "my markers", "health data", "health status",
    "what does my", "what are my", "how is my health", "am i healthy",
    "my fitness", "my training", "my recovery", "my body", "my vitals",
    # Bazi & metaphysics integration with health
    "bazi", "element", "recovery and bazi", "bazi guidance", "bazi advice",
]


def _is_health_query(query: str) -> bool:
    """Return True if the query is clearly health / medically oriented."""
    q = query.lower()
    return any(kw in q for kw in _HEALTH_QUERY_KEYWORDS)


def make_nodes(svc: OracleServices, db: OracleDB):
    profile_str = build_profile_str()

    def context_node(state: OracleState) -> dict:
        print("  [context] Fetching weather and lunar data...")
        meta    = svc.get_today_meta()
        weather = svc.get_weather()
        # Safely encode all parts to avoid Unicode issues on Windows
        safe_pillar = str(meta.get('day_pillar', '')).encode('ascii', 'ignore').decode('ascii')
        safe_officer = str(meta.get('officer_name', '')).encode('ascii', 'ignore').decode('ascii')
        safe_condition = str(weather.get('condition', '')).encode('ascii', 'ignore').decode('ascii')
        print(f"  {safe_pillar} Day | {safe_officer} Officer | {safe_condition}, {weather['temp_c']}C")

        garmin_ctx = None
        try:
            # Detect what time period user is asking about
            user_query = state.get("query", "")
            time_period = detect_time_period(user_query)

            # Get garmin data for the appropriate time period
            summary = db.get_garmin_summary(
                days=time_period["days"],
                start_date=time_period.get("start_date"),
                end_date=time_period.get("end_date"),
            )

            if summary["has_data"]:
                from garmin_sync import GarminSync, get_personalized_zones, analyze_training_vs_zones

                # Add time period context to the formatted summary
                _start = time_period.get("start_date", "")
                _end   = time_period.get("end_date", "")
                if _start and _end and _start != _end:
                    time_context = f"[{time_period['period'].upper()} DATA: {_start} to {_end}]"
                elif _start:
                    time_context = f"[{time_period['period'].upper()} DATA: {_start}]"
                else:
                    time_context = f"[{time_period['period'].upper()} DATA]"

                garmin_ctx = GarminSync.format_summary(
                    summary["activities"],
                    summary["latest_stats"],
                    stats_age_days=summary.get("stats_age_days"),
                    comprehensive=summary.get("comprehensive_health"),  # Pass comprehensive data if available
                )

                # Add WHO intensity minutes computed from activities (reliable, not from Garmin API)
                ci = summary.get("computed_intensity")
                if ci and (ci.get("moderate_minutes") or ci.get("vigorous_minutes")):
                    _mod = ci["moderate_minutes"]
                    _vig = ci["vigorous_minutes"]
                    _eq  = ci["who_equivalent"]
                    _gap = ci["who_target"] - _eq
                    _who_note = ("✅ meets WHO 150-min/week target" if _gap <= 0
                                 else f"⚠ {_gap} min-equivalent below WHO 150-min/week target")
                    _zone_note = " (actual zone data)" if any(
                        a.get("z1_minutes") is not None for a in summary["activities"]
                    ) else " (estimated from avg HR — re-sync for exact values)"
                    garmin_ctx += (
                        f"\n[WHO INTENSITY MINUTES — computed from this period's activities{_zone_note}]\n"
                        f"  Moderate (Z2, sub-VT1): {_mod} min\n"
                        f"  Vigorous (Z3–Z5, above VT1): {_vig} min\n"
                        f"  WHO-equivalent total (vigorous counts 2×): {_eq} min — {_who_note}\n"
                    )

                # Add zone analysis if training data available
                zones = get_personalized_zones()
                print(f"  [context] Zones loaded: VT1={zones.get('vt1_bpm')}, VT2={zones.get('vt2_bpm')}")
                if zones.get("vt1_bpm"):
                    # For multi-day queries, use appropriate time period in analysis
                    zone_analysis = analyze_training_vs_zones(summary["activities"], zones)
                    print(f"  [context] Zone analysis generated: {len(zone_analysis)} chars for {time_period['period']}")
                    if zone_analysis:
                        # Inject time period label into zone analysis
                        zone_analysis = f"\n{time_context}\n{zone_analysis}"
                        garmin_ctx += zone_analysis
                        print(f"  [context] Zone analysis ADDED to garmin_ctx")
                age_note = f", {time_period['period']} data"
                print(f"  [context] Garmin context loaded ({len(summary['activities'])} activities{age_note})")
        except Exception as _g_err:
            print(f"  [context] Garmin context unavailable: {_g_err}")

        lab_ctx = None
        try:
            lab_summary = db.get_lab_summary()
            if lab_summary["has_data"]:
                lab_ctx = _format_lab_context(lab_summary)
                print(f"  [context] Lab context loaded ({lab_summary['marker_count']} markers, "
                      f"{len(lab_summary['abnormal'])} flagged)")
        except Exception as _l_err:
            print(f"  [context] Lab context unavailable: {_l_err}")

        return {"meta": meta, "weather": weather,
                "garmin_context": garmin_ctx, "lab_context": lab_ctx}

    def classify_node(state: OracleState) -> dict:
        """
        CLASSIFIER AGENT: Routes queries to appropriate specialized agent.
        Uses local keyword matching with follow-up context awareness.
        """
        import time as timing
        t_start = timing.time()

        query       = state["query"]
        query_lower = query.lower()
        history     = state.get("history", [])
        # Preserve want_images=True if already set in initial state (from oracle_app.py)
        initial_wants_images = state.get("want_images", False)
        print(f"  [classify] Classifying intent... (initial want_images={initial_wants_images})")

        #  Step 1: Check for follow-up questions about the last response 
        # If the last assistant message was a recommendation, follow-up questions
        # about cost, hours, transport, reviews etc. should go to chat, not recommend.
        followup_triggers = [
            # Price / value
            "expensive", "cheap", "price", "cost", "how much", "value for money",
            "worth it", "affordable", "pricey",
            # Service / quality
            "service", "staff", "friendly", "kind", "courteous", "rude",
            "service quality", "customer service", "attentive", "helpful",
            "quality", "ambience", "atmosphere", "vibe", "interior",
            # Logistics
            "how far", "how do i get", "how long", "how to get",
            "mrt", "bus", "taxi", "grab", "parking", "near there", "nearby",
            "distance", "walking distance", "accessible",
            # Hours / booking
            "open now", "opening hours", "is it open", "what time", "closing time",
            "book", "reservation", "reserve", "walk in", "queue",
            # Food / menu
            "menu", "food", "drink", "must try", "signature", "specialty",
            "vegetarian", "vegan", "halal", "gluten",
            # Crowd / experience
            "crowded", "busy", "quiet", "noisy", "chill", "cosy", "cozy",
            "dress code", "wifi", "air con", "air conditioned",
            # General follow-up
            "is this", "is it", "tell me more", "more about", "what about",
            "sounds good", "sounds nice", "sounds great", "i like", "can i",
            "do they", "do they have", "are they", "is there", "what else",
            "any other", "alternative", "instead",
            "good reviews", "rating", "reviews", "recommended dish",
        ]
        last_assistant_was_recommend = False
        if history:
            recent_assistant = [m for m in history[-6:] if m.get("role") == "assistant"]
            if recent_assistant:
                last_msg = recent_assistant[-1].get("content", "")
                # Detect recommendation by multiple markers
                recommend_markers = [
                    "", "Recommended", "sector", "Officer", "Day Pillar",
                    "Singapore", "Bazi", "", "energy", "venue", "caf",
                    "restaurant", "park", "museum", "gallery", "hawker"
                ]
                if any(m in last_msg for m in recommend_markers):
                    last_assistant_was_recommend = True
                    print(f"  [classify] Step 1  last reply detected as recommendation")

        matched_followup = [t for t in followup_triggers if t in query_lower]
        if last_assistant_was_recommend and matched_followup:
            print(f"  [classify] [OK] Step 1 hit  follow-up keywords: {matched_followup[:3]}")
            print(f"  [classify] Intent: general_chat (follow-up about recommendation)")
            print(f"   Classify time: {timing.time() - t_start:.2f}s")
            return {"intent": "general_chat", "want_images": query_wants_images(query, initial_wants_images)}
        elif last_assistant_was_recommend:
            print(f"  [classify] Step 1  last was recommend but no followup keywords matched, continuing...")
        else:
            print(f"  [classify] Step 1  last reply was not a recommendation, continuing...")

        #  Step 1.5: Training analysis/optimality questions (chat, not SQL)
        # Questions about whether training was optimal, VO2 analysis, zone analysis, etc.
        # These should go to chat_node to use the zone analysis, not sql_node for raw data
        # BUT: Exclude queries asking for "summary", "total", "duration", "breakdown" of zones
        #      (those are data queries that need SQL, not analysis)
        analysis_triggers = [
            "optimal", "optimality", "efficient", "efficiency",
            "was my training", "is my training",
            "how was my workout", "how was my training",
            "vo2", "vo2 max", "anaerobic", "aerobic",
            "should i do less", "should i do more",
            "benefit", "benefits am i",
            "how is my", "how have i improved",
            "intensity zones", "training zones",
            "workout performance", "training performance",
            "last week", "this week", "this month",
            "progress", "improvement", "improving",
            # Health-status overview queries → chat_node (has full garmin context)
            "how am i doing", "how have i been", "how i've been",
            "body battery", "energy level", "recovery status",
            "am i recovering", "how recovered", "how is my recovery",
            "sleep quality", "how did i sleep", "how was my sleep",
            "training load", "training status", "training readiness",
            "overall health", "health status", "how healthy",
            "give me a", "full picture", "comprehensive",
            "what should i focus", "what should i do today",
            # Health/nutrition follow-ups (don't route to SQL)
            "vitamin", "supplement", "folate", "homocysteine",
            "b12", "b9", "b6", "nutrition", "diet", "nutrient",
            "deficiency", "marker", "lab result", "marker name",
            "blood work", "health marker", "elevated",
        ]
        # Keywords that indicate a raw data query (needs SQL), not LLM analysis
        data_query_keywords = ["how many", "much time", "spent in", "time in",
                               "what percentage", "show hours", "show percentages"]

        matched_analysis = [t for t in analysis_triggers if t in query_lower]
        has_data_query_keyword = any(k in query_lower for k in data_query_keywords)

        # Only route to chat if it's an analysis question without raw-data-only keywords
        if matched_analysis and not has_data_query_keyword:
            print(f"  [classify] Step 1.5 hit  analysis keywords: {matched_analysis[:2]}")
            print(f"  [classify] Intent: general_chat (analysis request)")
            print(f"   Classify time: {timing.time() - t_start:.2f}s")
            return {"intent": "general_chat", "want_images": query_wants_images(query, initial_wants_images)}
        elif has_data_query_keyword:
            print(f"  [classify] Step 1.5b detected data query keywords - routing to SQL instead")

        #  Step 2 (REORDERED): Recommendation requests FIRST (before SQL)
        # Check for venue/activity recommendations BEFORE checking for data queries
        # This prevents "suggest a place" or "activity" from being routed to SQL
        recommend_triggers = [
            "where should", "where can i", "suggest", "recommend",
            "should i go", "what to do", "what should i do",
            "find me", "take me", "bring me",
            "looking for a", "looking for somewhere",
            "good place for", "best place for",
            "coffee shop", "cafe", "restaurant", "food place", "dinner", "lunch", "breakfast",
            "yoga studio", "gym", "park", "museum", "gallery",
            "lonely", "alone", "isolated", "need company", "need people",
            "stressed", "anxious", "overwhelmed", "burnt out", "exhausted",
            "sad", "down", "depressed", "low mood", "blue",
            "bored", "boring", "nothing to do", "restless",
            "confused", "lost", "unclear", "need clarity", "stuck",
            "somewhere to", "a place to", "places to",
        ]
        matched_rec = [t for t in recommend_triggers if t in query_lower]
        if matched_rec:
            print(f"  [classify] Step 2 (PRIORITY) hit  recommend keywords: {matched_rec[:2]}")
            print(f"  [classify] Intent: recommend")
            print(f"   Classify time: {timing.time() - t_start:.2f}s")
            return {"intent": "recommend", "want_images": query_wants_images(query, initial_wants_images)}

        #  Step 3: SQL / memory queries
        sql_triggers = [
            "what did we discuss", "remind me", "show history", "show my",
            "which venues", "how many", "pattern", "last time", "visits",
            "conversation history", "what venues", "venues have we",
            "venues discussed", "previously discussed", "talked about",
            "discussed before", "what have we", "history of",
            "past conversations", "previous recommendations",
            # Garmin / movement / health queries
            "how have i been", "how am i doing", "how did i do",
            "how many steps", "how much did i", "how long did i",
            "what did i do", "what activities", "my activities",
            "my workouts", "my exercise", "this week", "last week",
            "recent activities", "recent workouts", "how active",
            "calories burned", "heart rate", "sleep score", "body battery",
            "moving this week", "how i've been", "training this week",
            # Additional health/fitness keywords (FIXED)
            "sleep", "training", "fitness", "recovery", "energy level",
            "workout", "exercise routine", "activity", "about my",
        ]
        matched_sql = [t for t in sql_triggers if t in query_lower]
        if matched_sql:
            print(f"  [classify] [OK] Step 3 hit  sql keywords: {matched_sql[:2]}")
            print(f"  [classify] Intent: sql_query")
            print(f"   Classify time: {timing.time() - t_start:.2f}s")
            return {"intent": "sql_query", "want_images": query_wants_images(query, initial_wants_images)}

        #  Step 4: RAG / profile queries 
        rag_triggers = [
            "who am i", "tell me about", "my chart", "my profile", "my bazi",
            "what am i", "about me", "my traits", "my characteristics"
        ]
        matched_rag = [t for t in rag_triggers if t in query_lower]
        if matched_rag:
            print(f"  [classify] [OK] Step 4 hit  rag keywords: {matched_rag[:2]}")
            print(f"  [classify] Intent: rag_query")
            print(f"   Classify time: {timing.time() - t_start:.2f}s")
            return {"intent": "rag_query", "want_images": query_wants_images(query, initial_wants_images)}

        #  Step 5: Activity + image (recommend with image flag)
        # "What activity suits the weather? Show me an image."
        # Must route to recommend (not image_node) so we get venue + activity first
        activity_triggers = ["activity", "activities", "what can i do", "what should i do today",
                             "what to do today", "suits the weather", "suits today"]
        if query_wants_images(query, initial_wants_images) and any(t in query_lower for t in activity_triggers):
            print(f"  [classify] [OK] Step 5 hit  activity+image combo  recommend with want_images=True")
            print(f"   Classify time: {timing.time() - t_start:.2f}s")
            return {"intent": "recommend", "want_images": True}

        #  Step 5b: Pure standalone image request (no recommend context)
        if query_wants_images(query, initial_wants_images):
            print(f"  [classify] Intent: image_request (pure/standalone)")
            print(f"   Classify time: {timing.time() - t_start:.2f}s")
            return {"intent": "image_request", "want_images": True}

        #  Step 6: LLM fallback classifier (only when no keyword matched) 
        # Uses Gemini Flash for fast, cheap classification (~0.5-1s).
        # Only reaches here for ambiguous queries that keywords couldn't resolve.
        print("  [classify] No keyword match  using LLM fallback classifier...")
        intent = _llm_classify(query, history, svc)
        print(f"  [classify] LLM intent: {intent}")
        print(f"   Classify time: {timing.time() - t_start:.2f}s")
        return {"intent": intent, "want_images": query_wants_images(query, initial_wants_images)}

    def structure_recommendation_prompt(query: str, venue_name: str, venue_summary: str,
                                       sector: str, meta: dict, weather: dict,
                                       garmin_context: str = "", lab_context: str = "",
                                       qwen_interpretation: dict = None) -> str:
        """
        PROMPT STRUCTURING LAYER: Enriches venue recommendation prompts with detailed context.

        Uses Qwen's interpretation (Tier 1) to provide better intent understanding,
        then enriches with Bazi, health, and weather context for DeepSeek (Tier 2).

        This extracts and structures:
        1. User intent from Qwen interpretation or query
        2. Current health state (body battery, stress, sleep)
        3. User profile (name, occupation, Bazi element)
        4. Today's Bazi energy
        5. Weather and time context
        6. What to emphasize (from Qwen)

        Returns a much richer prompt for DeepSeek reasoning.
        """

        if qwen_interpretation is None:
            qwen_interpretation = {}

        # Get intent from Qwen's interpretation, with fallback to query analysis
        intent = qwen_interpretation.get("intent", "venue recommendation")
        venue_type = qwen_interpretation.get("venue_type", "general")
        emphasis = qwen_interpretation.get("emphasis", "alignment with energy")
        must_avoid = qwen_interpretation.get("must_avoid", "")

        # Build health state summary from garmin_context
        health_state = "Unknown health state"
        if garmin_context:
            if "Body battery" in garmin_context or "battery" in garmin_context:
                health_state = "Low energy, needs recovery-focused activity"
            if "Stress" in garmin_context and "stress" in garmin_context.lower():
                health_state += " with elevated stress levels"

        # Weather and time context
        weather_detail = f"Current: {weather.get('condition', 'clear')}, {weather.get('temp_c', 25)}°C"
        is_rainy = weather.get("is_rainy", False)
        weather_note = "Currently rainy - recommend indoor or sheltered options" if is_rainy else "Weather is favorable for outdoor activities"

        # Bazi element balance explanation
        bazi_element = "Yang Fire"  # User's element
        today_element = meta.get('element_desc', 'unknown').split(':')[0]
        balance_note = f"Today's {today_element} energy requires balancing with Water/Earth elements to cool Yang Fire"

        # Build constraints section from Qwen
        constraints_section = ""
        if must_avoid:
            constraints_section = f"\nCONSTRAINTS:\n- Avoid: {must_avoid}"

        # Structure the prompt with Tier 1 insights
        structured_prompt = f"""
[TIER 1 INTERPRETATION - from Qwen 2.5 72B]
USER INTENT: {intent}
VENUE TYPE PREFERRED: {venue_type}
EMPHASIS IN RECOMMENDATION: {emphasis}
ORIGINAL QUERY: "{query}"
{constraints_section}

[CURRENT USER STATE]
- Name: Chen Ying Kai
- Bazi Element: Yang Wood Fire
- {health_state}
- {weather_detail}

[TODAY'S BAZI ENERGY]
- Day Pillar: {meta.get('day_pillar', 'Unknown')}
- 12 Officer: {meta.get('officer_name', 'Unknown')} ({meta.get('officer_meaning', '')})
- Auspicious Sector: {meta.get('auspicious_sector', 'Unknown')}
- Element Balance: {balance_note}

[WEATHER CONTEXT]
- {weather_note}
- Humidity: {weather.get('humidity', 'Unknown')}%
- UV Index: {weather.get('uv_index', '0')}

[VENUE DETAILS]
- Name: {venue_name}
- Sector: {sector}
- Category: {venue_type}
- Description: {venue_summary}

{"[HEALTH METRICS]" + garmin_context if garmin_context else ""}
{"[LAB NOTES]" + lab_context if lab_context else ""}

[TIER 2 REASONING TASK - for DeepSeek]
IMPORTANT: You MUST analyze ONLY the venue provided above: {venue_name}
Do NOT suggest alternative venues. Do NOT mention other places.
Focus entirely on explaining why {venue_name} fits the user's intent.

For {venue_name} only, explain:
1. Why this venue specifically matches the user's intent: {intent}
2. How {venue_name}'s location/atmosphere helps balance Fire with {today_element} (specifically {emphasis})
3. A specific activity the user can do AT THIS VENUE that supports their goal
4. Is this venue appropriate given their health state? (health_state: {health_state})
5. Any constraints to consider: {must_avoid if must_avoid else 'none'}

CONSTRAINTS FOR YOUR RESPONSE:
- Analyze ONLY {venue_name}. No alternative suggestions.
- Only mention details about {venue_name} that align with the venue description provided.
- Do NOT invent venue features that aren't described above.
- Format your response as two sections separated by "ACTIVITY:" — first your Bazi reasoning, then after "ACTIVITY:" a single sentence describing what the user should do at this venue.

Be specific, practical, and integrated with Bazi principles. Use concise language.
"""

        return structured_prompt

    def recommend_node(state: OracleState) -> dict:
        """
        RECOMMENDER AGENT: Generates venue recommendations with Bazi alignment.

        TIER 1+2 ARCHITECTURE:
        1. Tier 1 (Gemini 1.5 Flash): Interprets user intent and refines keywords
        2. Tier 2 (DeepSeek): Provides Bazi-aligned reasoning for recommendations
        """
        import time as timing
        start_total = timing.time()

        query       = state["query"]
        meta        = state["meta"]
        weather     = state["weather"]
        history_str = format_history(state.get("history", []))

        # Write debug log to file
        with open("debug_recommend.log", "a") as f:
            f.write(f"\n=== RECOMMEND CALLED ===\n")
            f.write(f"Query: {query}\n")

        print(f"  [recommend] Generating recommendations (Tier 1+2 architecture)... (want_images={state.get('want_images')})")

        weather_ctx = (
            f"{weather['condition']}, {weather['temp_c']}C, "
            f"humidity {weather['humidity']}%, UV {weather['uv_index']}"
        )
        meta_ctx = (
            f"Day Pillar: {meta['day_pillar']}, Energy: {meta['element_desc']}, "
            f"12 Officer: {meta['officer_name']}, QMDJ Sector: {meta['auspicious_sector']}"
        )

        # TIER 1: Use Gemini to interpret user intent and refine keywords
        t1 = timing.time()
        gemini_interpretation = svc.interpret_query_with_gemini(query, meta=meta, weather=weather, history=state.get("history", []))
        gemini_time = timing.time() - t1
        print(f"   Tier 1 interpretation (Gemini): {gemini_time:.2f}s", flush=True)
        with open("debug_recommend.log", "a") as f:
            f.write(f"[TIMING] Gemini intent: {gemini_time:.2f}s\n")

        # Extract refined keywords from Gemini's interpretation
        refined_kw = gemini_interpretation.get("refined_keywords", query)
        venue_type = gemini_interpretation.get("venue_type", "general")
        emphasis = gemini_interpretation.get("emphasis", "alignment with energy")

        # Write to debug log
        with open("debug_recommend.log", "a") as f:
            f.write(f"[GEMINI] venue_type='{venue_type}'\n")
            f.write(f"[GEMINI] refined_keywords='{refined_kw}'\n")

        # Use Gemini's refined keywords for search
        keywords = [k.strip() for k in refined_kw.split(",")]
        print(f"  [recommend] Tier 1 insights: intent='{gemini_interpretation.get('intent')}', type='{venue_type}', emphasis='{emphasis}'")

        t2 = timing.time()
        # Pass venue_type to search_places for API-level filtering
        places = svc.search_places(keywords, venue_type=venue_type)

        # Fallback: if refined keywords produced nothing, retry with the plain venue_type
        if not places and venue_type and venue_type != "general":
            print(f"  [recommend] Refined keywords returned 0 results — retrying with plain venue_type '{venue_type}'")
            with open("debug_recommend.log", "a") as f:
                f.write(f"[FALLBACK] Retrying with venue_type='{venue_type}'\n")
            places = svc.search_places([venue_type], venue_type=venue_type)
        _places_time = timing.time() - t2
        print(f"   Venue search (Google Places): {_places_time:.2f}s", flush=True)
        with open("debug_recommend.log", "a") as f:
            f.write(f"[TIMING] Places search: {_places_time:.2f}s\n")

        # Check for cancellation before reasoning
        if state.get("cancel_event") and state["cancel_event"].is_set():
            return {"response": "*Request cancelled.*", "history": state.get("history", [])}

        # Safely encode meta values to avoid Unicode issues on Windows
        safe_pillar_meta = str(meta.get('day_pillar', '')).encode('ascii', 'ignore').decode('ascii')
        safe_officer_meta = str(meta.get('officer_name', '')).encode('ascii', 'ignore').decode('ascii')

        print(f"\n{'='*65}")
        print(f"STRATEGIC ORACLE  |  {meta['date']}")
        print(f"{safe_pillar_meta} Day  |  {safe_officer_meta} Officer")
        print(f"{'='*65}\n")

        if not places:
            msg = (
                f"### No venues found for that query\n\n"
                f"The search for **{', '.join(keywords)}** returned no results in Singapore. "
                f"Try being more specific, e.g. *'suggest a caf'*, *'where to exercise'*, "
                f"or *'recommend a park'*.\n\n"
                f"*Today's energy: {meta['day_pillar']}  {meta['officer_name']} Officer  {weather_ctx}*"
            )
            # Still generate an energy image if requested
            if state.get("want_images"):
                elem = meta["element_desc"].split(":")[0]
                img_prompt = (
                    f"A person enjoying an outdoor activity in Singapore. "
                    f"Weather: {weather_ctx}. {elem} elemental energy. "
                    f"Vibrant, realistic scene."
                )
                svc.generate_image(img_prompt, f"activity_energy_{meta['date']}")
            return {"response": msg, "history": [{"role": "assistant", "content": msg}]}

        # Build shared context sections once (same for all venues)
        is_activity = _is_health_query(query) or any(
            kw in query.lower() for kw in ["activity", "workout", "exercise", "training", "fitness", "sport"]
        )
        garmin_section = ""
        if state.get("garmin_context"):
            framing = "Consider this when recommending intensity and recovery needs." if is_activity else \
                      "Use this to understand the user's current energy and recovery state when choosing a venue."
            garmin_section = f"\n{state['garmin_context']}\n{framing}\n"
            print("  [recommend] Garmin context injected")

        lab_section = ""
        if state.get("lab_context") and _is_health_query(query):
            lab_section = (
                f"\n{state['lab_context']}\n"
                f"Factor in any flagged markers when suggesting activity intensity or venue type.\n"
            )
            print("  [recommend] Lab context injected (health query detected)")

        if getattr(svc, "profile_source", "dict") == "pdf" and svc.chunks:
            _user_ctx = svc.chunks[0][:300]
            _rec_sys = (
                f"You are a Bazi and wellness advisor. The active user profile is:\n{_user_ctx}\n\n"
                "Analyze ONLY the venue provided in the user prompt. Do NOT suggest alternative venues. "
                "Focus on explaining why this specific venue matches their needs based on Bazi principles and current context. "
                "Do NOT invent venue features. Only use information provided."
            )
        else:
            _rec_sys = (
                "You are a Bazi and wellness advisor. "
                "Analyze ONLY the venue provided in the user prompt. Do NOT suggest alternative venues or make up venue names. "
                "Focus on explaining why THIS specific venue matches their needs based on Bazi principles and current context. "
                "Do NOT invent venue features that aren't described. Only analyze what is provided."
            )

        _default_insight = "This venue aligns with today's energy.\nACTIVITY: Spend time at this venue and soak in the atmosphere."

        def _reason_for_venue(p):
            name    = p.get("displayName", {}).get("text", "Unknown")
            address = p.get("formattedAddress", "")
            summary = p.get("generativeSummary", {}).get("overview", {}).get("text", "")
            sector  = p.get("_sector", "?")
            prompt  = structure_recommendation_prompt(
                query=query, venue_name=name, venue_summary=summary,
                sector=sector, meta=meta, weather=weather,
                garmin_context=garmin_section, lab_context=lab_section,
                qwen_interpretation=gemini_interpretation
            )
            t3 = timing.time()
            insight = svc.minimax(prompt, system=_rec_sys) or _default_insight
            print(f"   Reasoning for {name}: {timing.time() - t3:.2f}s", flush=True)
            return p, name, address, summary, sector, insight

        # Check for cancellation before the slow reasoning step
        if state.get("cancel_event") and state["cancel_event"].is_set():
            return {"response": "*Request cancelled.*", "history": state.get("history", [])}

        t3 = timing.time()
        venue_results = [_reason_for_venue(places[0])]
        print(f"   Reasoning (1 venue): {timing.time() - t3:.2f}s", flush=True)

        full_response = []
        for p, name, address, summary, sector, full_insight in venue_results:

            # Split insight from activity description
            insight      = full_insight
            activity_desc = f"Visit {name} and enjoy what it has to offer"
            if "ACTIVITY:" in full_insight:
                parts        = full_insight.split("ACTIVITY:", 1)
                insight      = parts[0].strip()
                activity_desc = parts[1].strip()

            # Safe encoding for Windows terminal (charmap support)
            safe_name = str(name).encode('ascii', 'ignore').decode('ascii')
            safe_address = str(address).encode('ascii', 'ignore').decode('ascii')
            safe_sector = str(sector).encode('ascii', 'ignore').decode('ascii')
            safe_insight = str(insight).encode('ascii', 'ignore').decode('ascii')
            safe_activity = str(activity_desc).encode('ascii', 'ignore').decode('ascii')

            print(f"\nPLACE:    {safe_name}\n          {safe_address}\n          {safe_sector} Sector")
            print(f"\nINSIGHT:  {safe_insight}")
            print(f"\nACTIVITY: {safe_activity}\n" + "-" * 65)

            # Generate image AFTER we know the specific activity  much more relevant
            if state.get("want_images"):
                weather_cond = str(weather.get('condition', 'clear')).encode('ascii', 'ignore').decode('ascii')
                weather_desc = "rainy and cosy" if weather.get("is_rainy") else f"{weather['temp_c']}C {weather_cond.lower()}"
                activity_img_prompt = (
                    f"A person {activity_desc} at {name} in Singapore. "
                    f"Weather is {weather_desc}. "
                    f"{summary[:120] if summary else ''}. "
                    f"Photorealistic, vibrant, immersive scene, cinematic lighting."
                )
                safe_prompt = str(activity_img_prompt[:120]).encode('ascii', 'ignore').decode('ascii')
                print(f"  [image] Prompt: {safe_prompt}...")
                svc.generate_image(activity_img_prompt, f"activity_{safe_filename(name)}")

            # Build markdown-formatted response for clean Streamlit rendering
            # Safe encoding for markdown display - MUST sanitize all parts
            safe_name_md = str(name).encode('ascii', 'ignore').decode('ascii')
            safe_address_md = str(address).encode('ascii', 'ignore').decode('ascii')
            safe_insight_md = str(insight).encode('ascii', 'ignore').decode('ascii')
            safe_pillar = str(meta.get('day_pillar', '')).encode('ascii', 'ignore').decode('ascii')
            safe_officer = str(meta.get('officer_name', '')).encode('ascii', 'ignore').decode('ascii')
            safe_weather = str(weather_ctx).encode('ascii', 'ignore').decode('ascii')

            md = (
                f"###  {safe_name_md}\n"
                f"**{safe_address_md}**  {safe_sector} Sector\n\n"
                f"{safe_insight_md}\n\n"
                f"---\n"
                f"*{safe_pillar} Day  {safe_officer} Officer  "
                f"{safe_weather}*"
            )
            # Ensure the final markdown is also ASCII-safe
            md_safe = md.encode('ascii', 'ignore').decode('ascii')
            full_response.append(md_safe)

            # Save venue to venue_visits table using correct OracleDB method
            try:
                db.log_venue_visit(
                    venue_name=name,
                    venue_address=address,
                    venue_sector=sector,
                    day_pillar=meta.get("day_pillar"),
                    officer_type=meta.get("officer_name"),
                    weather=weather,
                )
                print(f"  [OK] Venue saved to DB: {name}")
            except Exception as _db_err:
                print(f"  [WARNING] Venue DB save failed: {_db_err}")

        response = "\n\n".join(full_response)
        # Final safety pass to ensure no Unicode in response
        response_safe = response.encode('ascii', 'ignore').decode('ascii')
        _total = timing.time() - start_total
        print(f"\n TOTAL TIME: {_total:.2f}s\n", flush=True)
        with open("debug_recommend.log", "a") as f:
            f.write(f"[TIMING] TOTAL: {_total:.2f}s\n")
        return {"response": response_safe, "history": [{"role": "assistant", "content": response_safe}]}

    def rag_node(state: OracleState) -> dict:
        query       = state["query"]
        history_str = format_history(state.get("history", []))
        print("  [rag] Searching profile document...")

        chunks      = svc.retrieve_chunks(query, top_k=4)
        chunks_text = "\n---\n".join(chunks)

        # Detect active profile source to set correct system context
        profile_source = getattr(svc, "profile_source", "dict")
        if profile_source == "pdf":
            # Neutral system prompt  let the PDF chunks speak for themselves
            rag_system = (
                "You are a personal advisor helping answer questions about a user's profile. "
                "Answer ONLY from the profile excerpts provided. "
                "Do not reference or assume any prior profile  use only what is in the excerpts. "
                f"{STYLE_GUARD}"
            )
        else:
            # Default: use Chen Yingkai Bazi context
            rag_system = BAZI_SYSTEM_PROMPT

        prompt = (
            f"Answer the question using ONLY the profile excerpts below. "
            f"Do not use any prior knowledge about other people.\n\n"
            f"Question: {query}\n\n"
            f"Profile excerpts:\n{chunks_text}\n\n"
            f"Answer in 3-5 sentences, grounded in the profile text."
        )
        answer = svc.deepseek(prompt, system=rag_system) or "Not found in profile."

        print(f"\n{'='*65}\nDOCUMENT INSIGHT [{profile_source.upper()}]\n{'='*65}\n{answer}\n" + "-"*65)
        return {"response": answer, "history": [{"role": "assistant", "content": answer}]}

    def chat_node(state: OracleState) -> dict:
        query       = state["query"]
        query_lower = query.lower()
        meta        = state["meta"]
        history     = state.get("history", [])
        history_str = format_history(history)
        print("  [chat] Responding conversationally...")

        # Check if this is a follow-up about a recent recommendation
        recent_assistant = [m for m in history[-4:] if m.get("role") == "assistant"]
        last_recommendation = ""
        if recent_assistant:
            last_msg = recent_assistant[-1].get("content", "")
            if "" in last_msg or "sector" in last_msg.lower():
                last_recommendation = f"\n\nMost recent recommendation:\n{last_msg[:600]}"

        # Use PDF profile context if a PDF is loaded
        if getattr(svc, "profile_source", "dict") == "pdf":
            _profile_context = "\n".join(svc.chunks[:2]) if svc.chunks else "No profile loaded."
            _chat_system = (
                "You are a warm personal advisor. Answer based on the user profile provided. "
                f"{STYLE_GUARD}"
            )
        else:
            _profile_context = profile_str
            _chat_system = "Warm Singapore advisor. " + STYLE_GUARD

        # Only inject Garmin context for activity/health queries
        garmin_section = ""
        asks_about_bazi_recovery = any(kw in query.lower() for kw in ["bazi", "recovery", "how am i doing", "energy level"])

        if state.get("garmin_context") and (
            _is_health_query(query) or
            any(kw in query.lower() for kw in [
                "activity", "workout", "exercise", "training", "fitness", "sport",
                "moving", "active", "fared", "fare", "this week", "last week",
                "past week", "run", "ran", "swim", "cycle", "walk", "pickleball",
                "weekly", "week", "how did i", "how have i",
            ])
        ):
            garmin_section = f"\nUser's recent Garmin health data:\n{state['garmin_context']}\n"
            print("  [chat] Garmin context injected (activity/health query detected)")
        elif asks_about_bazi_recovery and not state.get("garmin_context"):
            # Special handling: user asked about Bazi/recovery but no Garmin context available
            garmin_section = (
                "\n[NO RECOVERY DATA SYNCED YET]\n"
                "Your device's sleep, HRV, body battery, and training readiness data haven't synced to the database yet.\n"
            )
            print("  [chat] Bazi/recovery query detected but Garmin data unavailable")

        # Only inject lab context when the query is explicitly health-related
        lab_section = ""
        if state.get("lab_context") and _is_health_query(query):
            lab_section = f"\nUser's lab health markers:\n{state['lab_context']}\n"
            print("  [chat] Lab context injected (health query detected)")

        lab_instruction = (
            "If the user asks about their lab results, blood markers, or health risks, "
            "refer specifically to the lab data above. "
            if lab_section else ""
        )

        # ALWAYS inject Bazi context for Bazi/recovery queries (even if not health_overview)
        if asks_about_bazi_recovery and meta and meta.get('personal_impact'):
            bazi_context = (
                f"\n[TODAY'S BAZI CONTEXT — How Today Affects Your Day Master 丙 (Yang Fire)]\n"
                f"  Today's Stem: {meta.get('day_quality', 'unknown')}\n"
                f"  Impact on your profile: {meta.get('personal_impact', '')}\n"
                f"  Element colour: {meta.get('quality_colour', '')}\n"
                f"  What to embrace: {meta.get('do_today', '')}\n"
                f"  What to avoid: {meta.get('avoid_today', '')}\n"
                f"  Training & activity guidance: {meta.get('training_advice', '')}\n"
                f"  Mind & psyche note: {meta.get('psyche_note', '')}\n"
                f"  Officer energy type: {meta.get('officer_energy', '')}\n"
                f"  How it affects you: {meta.get('officer_personal', '')}\n"
                f"  Your response strategy: {meta.get('officer_advice', '')}\n"
            )
            if garmin_section and "[TODAY'S BAZI CONTEXT" not in garmin_section:
                garmin_section += bazi_context
                print("  [chat] Bazi context injected into Garmin section")
            elif not garmin_section:
                # If no garmin section, add Bazi as separate context
                garmin_section = bazi_context
                print("  [chat] Bazi context set as primary context")

        # Special instruction for Bazi/recovery queries
        bazi_recovery_instruction = ""
        if asks_about_bazi_recovery:
            if state.get("garmin_context") and "[DAILY SNAPSHOT" in state.get("garmin_context", ""):
                # Recovery data available
                bazi_recovery_instruction = (
                    "\n[INSTRUCTION FOR BAZI+RECOVERY QUERY]\n"
                    "User is asking about activity based on BOTH recovery metrics and Bazi element.\n"
                    "You MUST address both factors:\n"
                    "1. Their physical state: sleep quality, HRV, body battery, resting HR, stress\n"
                    "2. Their elemental state: today's stem + branch, how it interacts with their Fire profile\n"
                    "Recommend a specific activity that respects BOTH frameworks.\n"
                    "Explain the DOUBLE benefit: why this activity helps physiologically AND metaphysically.\n"
                )
            else:
                # Recovery data not available, use Bazi alone
                bazi_recovery_instruction = (
                    "\n[INSTRUCTION FOR BAZI+RECOVERY QUERY — NO RECOVERY DATA YET]\n"
                    "User asked about activity based on recovery + Bazi, but recovery data hasn't synced.\n"
                    "TELL THEM:\n"
                    "Your recovery stats (sleep, HRV, body battery) need to sync from your Garmin device.\n"
                    "1. Open the app sidebar and click 🔄 RE-SYNC GARMIN\n"
                    "2. Complete the login and sync process\n"
                    "3. Wait 15–30 minutes for your Garmin device to send data\n"
                    "4. Ask again: 'What activity should I do today based on my recovery and Bazi?'\n\n"
                    "BUT MEANWHILE, based on today's Bazi element alone:\n"
                    "Use the [TODAY'S BAZI CONTEXT] above to recommend an activity that aligns with:\n"
                    "- Today's stem energy (what element dominates today)\n"
                    "- How it feeds, controls, or exhausts their Fire profile\n"
                    "- Their personal 'What to Embrace' and 'What to Avoid' for today\n"
                    "- The 'Activity Guidance' for today's element\n"
                    "Be specific: 'Today's Wood element suggests growth-focused activities like...' etc.\n"
                )

        zone_instruction = ""
        if state.get("garmin_context") and "[ZONE PRESCRIPTION" in state.get("garmin_context", ""):
            # Get personalized zones to include in instruction
            from garmin_sync import get_personalized_zones
            zones = get_personalized_zones()
            vt1 = zones.get('vt1_bpm', 156)
            vt2 = zones.get('vt2_bpm', 173)

            zone_instruction = (
                f"*** OVERRIDE: The Garmin data includes a detailed [ZONE PRESCRIPTION] section. "
                f"THIS TAKES PRECEDENCE over the usual response structure. "
                f"You MUST output the complete [ZONE PRESCRIPTION] section VERBATIM, exactly as provided. "
                f"Do NOT reformat it, summarize it, or condense it. "
                f"After the prescription, briefly add the activity log and key metrics, but the prescription is the priority. "
                f"The user specifically wants to see per-zone role, impact (too little / too much), targets, and status for ALL zones (Z1–Z5). "
                f"Quote exact numbers, targets, and consequence statements from the prescription directly. "
                f"Do NOT paraphrase — preserve the exact wording. ***\n\n"
            )
            print(f"  [chat] *** ZONE INSTRUCTION ADDED - Zone ranges: Z1:<{vt1}, Z2:{vt1}-{vt2}, Z3+:>{vt2} ***")

        is_health_overview = garmin_section and not any(kw in query_lower for kw in [
            "this week", "last week", "past week", "fared", "weekly review",
            "week recap", "week summary", "how did i fare", "how did i do this week",
        ]) and any(kw in query_lower for kw in [
            "how am i doing", "how have i been", "full picture", "comprehensive",
            "body battery", "overall health", "health status", "training readiness",
            "how am i", "doing today", "state today", "today's activities",
            "today", "this morning", "this evening",
        ])

        is_weekly_review = garmin_section and any(kw in query_lower for kw in [
            "this week", "last week", "past week", "how did i fare", "how did i do",
            "weekly recap", "week recap", "week summary", "weekly review",
            "how was my week", "how have i done", "fared", "previous week",
        ])

        if is_health_overview:
            # Check if Bazi context was already injected (universal Bazi injection above)
            _has_recovery_data = state.get("garmin_context") and "[DAILY SNAPSHOT" in state.get("garmin_context", "")
            _has_bazi_context = garmin_section and "[TODAY'S BAZI CONTEXT" in garmin_section

            if _has_recovery_data and _has_bazi_context:
                # Full health overview instruction (both recovery + Bazi data available)
                garmin_instruction = (
                    "You are a sports and mental health physician with expertise in Chinese metaphysics, "
                    "with 20 years of clinical experience. Your role is to integrate two frameworks:\n"
                    "1. **Clinical physiology** (recovery metrics, training load, ANS state)\n"
                    "2. **Bazi metaphysics** (how today's elemental energy interacts with the user's Fire profile)\n\n"
                    "Use ONLY the Garmin data and Bazi context explicitly shown above. "
                    "Do NOT invent, estimate, or assume any values not present. "
                    "When both frameworks suggest a direction (e.g. low HRV + Bazi avoid strain = REST), "
                    "name both reasons.\n\n"
                    "Structure your response in exactly four sections:\n\n"
                    "**1. Yesterday's Recovery — How You're Starting Today**\n"
                    "Interpret each available indicator as a physician would on a morning ward round. "
                    "For every metric present, state the value AND what it means clinically:\n"
                    "- Body battery at wake (energy reserve — below 50 means high fatigue risk)\n"
                    "- Sleep: total hours + score + stages if available\n"
                    "- HRV: value, status, and trend — your primary ANS and recovery readiness marker\n"
                    "- Resting HR, breathing rate, stress score\n"
                    "- Training readiness score if available\n"
                    "End with: GREEN (push) / AMBER (moderate) / RED (recover) from a physiology perspective?\n\n"
                    "**2. Today's Bazi Context — Element Interaction with Your Fire Profile**\n"
                    "- TODAY'S ELEMENT: dominant stem energy and its effect on your 丙 (Yang Fire) profile\n"
                    "- INTERACTION: How does it feed, control, or exhaust your Fire?\n"
                    "- CONVERGENCE: Where do physiology and metaphysics align or diverge?\n\n"
                    "**3. Today's Activity (if any)**\n"
                    "If activity logged today: what was it, duration, HR zone, alignment with recovery + Bazi?\n\n"
                    "**4. Today's Prescription — Integrated Recovery + Bazi Wisdom**\n"
                    "- What to do next: specific activity/recovery, duration, HR zone\n"
                    "- WHY: explain physiological mechanism + metaphysical alignment (double benefit)\n"
                    "- What to avoid and why (physiology + Bazi reasons)\n"
                    "Write as a physician who respects both science and wisdom."
                )
            elif _has_bazi_context and not _has_recovery_data:
                # Fallback instruction (recovery data missing, Bazi available)
                garmin_instruction = (
                    "Your recovery data (sleep, HRV, body battery) hasn't synced from Garmin yet, but we have "
                    "your Bazi element context for today.\n\n"
                    "TELL THE USER:\n"
                    "Your recovery stats (sleep, HRV, body battery) haven't synced yet. "
                    "To get full recovery + Bazi integration:\n"
                    "1. Tap 🔄 RE-SYNC GARMIN at the top of the sidebar\n"
                    "2. Sign in and complete the sync\n"
                    "3. Wait 15–30 minutes for your Garmin device to send data to their servers\n"
                    "4. Ask again: 'How am I doing today based on recovery and Bazi?'\n\n"
                    "BUT MEANWHILE, based on today's Bazi context alone (see [TODAY'S BAZI CONTEXT] above):\n"
                    "Explain how today's element guides activity choice and mindset, "
                    "even without physiological recovery metrics.\n\n"
                    "Once recovery data syncs, you'll get the full integration of physiology + metaphysics."
                )
            else:
                # Neither recovery nor Bazi data available (shouldn't happen, but fallback)
                garmin_instruction = (
                    "Data is limited. Suggest the user re-sync Garmin to get recovery metrics, "
                    "then ask again for personalized recovery + Bazi guidance."
                )

            effective_lab_section = ""
            effective_lab_instruction = ""
        elif is_weekly_review:
            _is_last_week = any(kw in query_lower for kw in ["last week", "previous week", "prior week"])
            _week_label = "Last Week (previous calendar week, Mon–Sun)" if _is_last_week else "This Week (current calendar week, Mon–today)"
            garmin_instruction = (
                "You are a sports physician and performance coach with 20 years of experience. "
                "Use ONLY the Garmin data explicitly shown above. "
                "CRITICAL: Do NOT invent, hallucinate, or assume any activities, metrics, or values "
                "not present in the data. If the data shows only 1 activity this week, report exactly 1 activity — "
                "do not fabricate additional sessions. If a metric (sleep, HRV, etc.) is absent, say 'not recorded'. "
                "Reference only specific numbers and dates that appear in the data above.\n\n"
                f"The data covers: **{_week_label}**. "
                "State the date range explicitly at the start of your response.\n\n"
                "Structure your response in three sections:\n\n"
                f"**{_week_label} — Activity Log**\n"
                "List EVERY session in the data: date, activity type, duration, avg HR, distance if available. "
                "Classify each by VO2 Master zone (Z1 <140bpm / Z2 140–156 / Z3 156–164 / Z4 164–173 / Z5 >173) "
                "if heart rate is available. Calculate total weekly training volume (minutes and hours).\n\n"
                "**Weekly Training Assessment**\n"
                "Evaluate the week's training:\n"
                "- Zone distribution (count and minutes per zone, as % of total)\n"
                "- Zone 2 volume: is it meeting the 3–4 hour/week aerobic base target?\n"
                "- Load balance verdict: overreaching / optimal / under-stimulated?\n"
                "- Recovery signals from daily stats: sleep quality trend, HRV, resting HR, body battery\n"
                "- Weekly intensity minutes vs WHO target (150 min moderate + 75 vigorous)\n"
                "Give an explicit overall verdict: OPTIMAL / GOOD / SUBOPTIMAL — and why.\n\n"
                "**Prescription for the Coming Week**\n"
                "- Specific sessions to add or reduce with duration and HR zone\n"
                "- Recovery day recommendations if cumulative load is high\n"
                "- One targeted VO2 improvement session with exact duration, HR target, and the physiological mechanism\n"
                "Write this as a coach issuing a training plan — specific dates, specific actions."
            )
            effective_lab_section = ""
            effective_lab_instruction = ""
        else:
            garmin_instruction = (
                "IMPORTANT: Use ONLY the Garmin data explicitly provided above — activities, daily stats, "
                "HRV, training readiness, body battery, sleep, stress, and comprehensive metrics. "
                "Do NOT invent, estimate, or assume any values not shown. "
                "Do NOT say 'limited data' if any fields are present — interpret what is available. "
                "Reference specific values and dates from the data. "
            )
            effective_lab_section = lab_section
            effective_lab_instruction = lab_instruction

        response_length = (
            ""  # prompt structure handled inside garmin_instruction for health overview
        ) if is_health_overview else "Respond in 2-4 sentences, warm and direct."

        # FORCE Bazi context in prompt for Bazi/recovery queries (if not already in garmin_section)
        direct_bazi_section = ""
        if asks_about_bazi_recovery and "[TODAY'S BAZI CONTEXT" not in garmin_section and meta.get('personal_impact'):
            direct_bazi_section = (
                f"\n[TODAY'S BAZI CONTEXT — Personalized for Your Day Master 丙 (Yang Fire)]\n"
                f"Today's Stem Energy: {meta.get('day_quality', 'unknown')}\n"
                f"Personal Impact: {meta.get('personal_impact', '')}\n"
                f"What to Embrace Today: {meta.get('do_today', '')}\n"
                f"What to Avoid: {meta.get('avoid_today', '')}\n"
                f"Activity Guidance: {meta.get('training_advice', '')}\n"
                f"Mind & Psyche: {meta.get('psyche_note', '')}\n"
                f"Officer Energy: {meta.get('officer_energy', '')}\n"
                f"Personal Officer Impact: {meta.get('officer_personal', '')}\n"
                f"Your Response Strategy: {meta.get('officer_advice', '')}\n"
            )
            print("  [chat] Direct Bazi section added to prompt")

        prompt = (
            f"{STYLE_GUARD}\n\nProfile: {_profile_context[:600]}\n\n"
            f"Today: {meta['element_desc']} | {meta['officer_name']}\n"
            f"{direct_bazi_section}"
            f"\n{garmin_section}"
            f"{effective_lab_section}"
            f"Conversation history:\n{history_str}"
            f"{last_recommendation}\n\n"
            f"User follow-up: \"{query}\"\n\n"
            f"{bazi_recovery_instruction}"
            f"{zone_instruction}"
            f"{garmin_instruction}"
            f"If the user asks about their high-intensity activities or recent workouts, "
            f"refer specifically to the Garmin data above (activity names, dates, HR, duration). "
            f"{effective_lab_instruction}"
            f"If the user is asking about the recommended venue above, answer specifically about that place. "
            f"Use your general knowledge about Singapore venues. "
            f"{response_length}"
        )
        answer = svc.deepseek(prompt, system=_chat_system) or "Tell me more."
        print(f"\n{answer}\n")
        return {"response": answer, "history": [{"role": "assistant", "content": answer}]}

    def image_node(state: OracleState) -> dict:
        meta = state["meta"]
        elem = meta["element_desc"].split(":")[0]
        print(f"  [image] Generating {elem} energy image (Nano Banana primary)...")
        path = svc.generate_image(f"{elem} energy. {meta['visual']}.", f"energy_{meta['date']}")
        msg  = f"Image saved: {path}" if path else "Image generation failed."
        print(f"\n{msg}\n")
        return {"response": msg, "history": [{"role": "assistant", "content": msg}]}

    return {
        "context":   context_node,
        "classify":  classify_node,
        "recommend": recommend_node,
        "rag":       rag_node,
        "chat":      chat_node,
        "image":     image_node,
    }


# =============================================================================
# LANGGRAPH ROUTER (with SQL)
# =============================================================================
def route_intent(state: OracleState) -> str:
    intent = state.get("intent", "recommend")
    return {
        # Keyword classifier intents
        "recommend":     "recommend",
        "rag_query":     "rag",
        "image_request": "image",
        "general_chat":  "chat",
        "sql_query":     "sql",
        # LLM classifier intents (shorter labels)
        "chat":          "chat",
        "rag":           "rag",
        "image":         "image",
        "sql":           "sql",
    }.get(intent, "recommend")


# =============================================================================
# GRAPH BUILDER
# =============================================================================
def build_graph(svc: OracleServices, db: OracleDB):
    nodes = make_nodes(svc, db)
    graph = StateGraph(OracleState)

    graph.add_node("context",   nodes["context"])
    graph.add_node("classify",  nodes["classify"])
    graph.add_node("recommend", nodes["recommend"])
    graph.add_node("rag",       nodes["rag"])
    graph.add_node("chat",      nodes["chat"])
    graph.add_node("image",     nodes["image"])
    
    sql_node = make_sql_node(svc, db)
    graph.add_node("sql", sql_node)

    graph.set_entry_point("context")
    graph.add_edge("context", "classify")

    graph.add_conditional_edges(
        "classify",
        route_intent,
        {
            "recommend": "recommend",
            "rag":       "rag",
            "chat":      "chat",
            "image":     "image",
            "sql":       "sql",
        }
    )

    graph.add_edge("recommend", END)
    graph.add_edge("rag",       END)
    graph.add_edge("chat",      END)
    graph.add_edge("image",     END)
    graph.add_edge("sql",       END)

    return graph.compile()


# =============================================================================
# CONVERSATIONAL LOOP
# =============================================================================
def run_conversation():
    print("\n" + "="*70)
    print("  STRATEGIC ORACLE  Optimized Multi-Agent System")
    print(f"  Profile: {HARDCODED_PROFILE['name']}")
    print("  Architecture: LangGraph Controller + 6 Specialized Agents")
    print("  Classification: Local (instant, profile-aware)")
    print("  Keywords: Local extraction (context + Bazi + weather aware)")
    print("  Reasoning: DeepSeek (high quality Bazi insights)")
    print("  Image Gen: Nano Banana (gemini-2.5-flash-image)  Pollinations (free fallback)")
    print("  Memory: SQLite (persistent storage)")
    print("  Commands: 'quit' | 'show graph' | 'db stats'")
    print("="*70 + "\n")

    svc      = OracleServices()
    db       = OracleDB("oracle.db")
    compiled = build_graph(svc, db)
    history  = []

    logged_graph = wrap_with_logging(compiled, db)

    try:
        print("Graph structure:")
        print(compiled.get_graph().draw_mermaid())
        print()
    except Exception:
        pass

    print("Oracle ready. Local intelligence + DeepSeek reasoning + Gemini images!\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nFarewell. May your path be auspicious.")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "bye"):
            print("\nFarewell. May your path be auspicious.")
            break
        if query.lower() == "show graph":
            try:
                print(compiled.get_graph().draw_mermaid())
            except Exception as e:
                print(f"Unavailable: {e}")
            continue
        if query.lower() == "db stats":
            stats = db.get_stats()
            print(f"\nDatabase Statistics (oracle.db):")
            for table, count in stats.items():
                print(f"  {table}: {count} records")
            continue

        history.append({"role": "user", "content": query})

        initial_state: OracleState = {
            "query":       query,
            "intent":      None,
            "meta":        None,
            "weather":     None,
            "history":     history,
            "response":    None,
            "want_images": False,
        }

        try:
            result = logged_graph.invoke(initial_state)
            history = result.get("history", history)
            if len(history) > MAX_HISTORY:
                history = history[-MAX_HISTORY:]
        except Exception as e:
            print(f"\n[WARNING] Error processing query: {e}")
            print("Please try rephrasing or check your API keys.\n")
            continue

        print()


# =============================================================================
# RUN
# =============================================================================
if __name__ == "__main__":
    run_conversation()
