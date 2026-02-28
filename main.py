from fastapi import FastAPI
from pydantic import BaseModel
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import os

app = FastAPI()

# Ensure vader lexicon exists (safe check)
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

sia = SentimentIntensityAnalyzer()

class Comment(BaseModel):
    comment: str

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