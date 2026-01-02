"""
ai_clients.py
---------------------------------
Centralized AI client initialization for:
1) OpenAI (GPT-4o / GPT-4o-mini – vision + text)
2) Google Gemini (image editing / generation)

This file MUST be imported by other modules.
DO NOT initialize clients elsewhere.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai

# =========================================================
# 🔹 LOAD ENV VARIABLES (.env for local, Render env for prod)
# =========================================================

load_dotenv()  # Safe on Render, required locally

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("❌ OPENAI_API_KEY is not set")

if not GEMINI_API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY is not set")

# =========================================================
# 🔹 OPENAI CLIENT (GPT-4o / GPT-4o-mini)
# =========================================================

openai_client = OpenAI(
    api_key=OPENAI_API_KEY
)

# =========================================================
# 🔹 GEMINI CLIENT (IMAGE MODEL)
# =========================================================

genai.configure(api_key=GEMINI_API_KEY)

GEMINI_IMAGE = genai.GenerativeModel(
    model_name="models/gemini-2.5-flash-image"
)

# =========================================================
# 🔹 OPTIONAL: SIMPLE HEALTH CHECK
# =========================================================

def health_check():
    """
    Quick sanity check to ensure clients are loaded.
    """
    return {
        "openai": "ready",
        "gemini": "ready"
    }
