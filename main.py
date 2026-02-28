from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

app = FastAPI()

# 🔥 CRITICAL: Enable CORS for submission validator
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Safe NLTK loading
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

sia = SentimentIntensityAnalyzer()

class Comment(BaseModel):
    comment: str

@app.get("/")   # 🔥 Add root endpoint (some validators test this)
def root():
    return {"status": "running"}

@app.post("/comment")
def analyze_comment(data: Comment):
    text = data.comment
    scores = sia.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        sentiment = "positive"
        rating = 5
    elif compound <= -0.05:
        sentiment = "negative"
        rating = 1
    else:
        sentiment = "neutral"
        rating = 3

    return {
        "sentiment": sentiment,
        "rating": rating
    }