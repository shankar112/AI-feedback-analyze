from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class FeedbackRequest(BaseModel):
    feedback: str


class FeedbackAnalysis(BaseModel):
    overall_sentiment: Literal["Positive", "Negative", "Neutral"]
    positive_words: list[str] = Field(default_factory=list)
    negative_words: list[str] = Field(default_factory=list)
    neutral_words: list[str] = Field(default_factory=list)
    auto_reply_email: str


app = FastAPI(title="AI Feedback Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/analyze-feedback", response_model=FeedbackAnalysis)
def analyze_feedback(payload: FeedbackRequest) -> FeedbackAnalysis:
    feedback_text = payload.feedback.strip()
    if not feedback_text:
        raise HTTPException(status_code=400, detail="The 'feedback' field must not be empty.")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Missing GEMINI_API_KEY in .env file.")

    prompt = (
        "Analyze the customer feedback and return structured JSON. "
        "Classify the overall sentiment as Positive, Negative, or Neutral. "
        "Extract words or short phrases into positive_words, negative_words, and neutral_words. "
        "Generate a concise professional auto-reply email based on the feedback.\n\n"
        f"Feedback:\n{feedback_text}"
    )

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=FeedbackAnalysis,
            ),
        )

        if not response.parsed:
            raise HTTPException(status_code=500, detail="Gemini returned an empty analysis.")

        return response.parsed
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to analyze feedback: {exc}") from exc
