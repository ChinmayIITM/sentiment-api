from fastapi import FastAPI
from pydantic import BaseModel
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon (required for cloud deployment like Render)
nltk.download("vader_lexicon")

app = FastAPI()

# Initialize sentiment analyzer AFTER downloading lexicon
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