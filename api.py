"""
The backend: a small FastAPI service that analyzes ONE customer review.

Run it with:
    fastapi dev api.py

Then the Streamlit app (app.py) will call this service for every review.
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field
import ollama


app = FastAPI()


# What the caller must SEND us
class Review(BaseModel):
    text: str


# What Llama gives back, and what we SEND to the caller.
class Analysis(BaseModel):
    label: str
    score: int = Field(ge=1, le=5)
    theme: str


@app.post("/analyze")
def analyze(review: Review):

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": (
                    "Analyze this customer review.\n"
                    "label must be 'positive', 'negative', or 'neutral'.\n"
                    "score must be a number from 1 (very bad) to 5 (very good).\n"
                    "theme must be ONE lowercase word for the main topic "
                    "(for example: delivery, taste, price, service, quality).\n"
                    f"Review: {review.text}"
                ),
            }
        ],
        format=Analysis.model_json_schema(),
    )

    return Analysis.model_validate_json(
        response["message"]["content"]
    )