# sentiment_analyzer/app.py

from flask import Flask, request, jsonify
from transformers import pipeline
from pymongo import MongoClient
from datetime import datetime
import plotly.express as px
import pandas as pd
import os

# Initialize Flask app
app = Flask(__name__)

# Load sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client["sentimentDB"]
collection = db["reviews"]

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("text")

    if not text:
        return jsonify({"error": "No text provided."}), 400

    result = sentiment_pipeline(text)[0]
    sentiment = result["label"].lower()
    confidence = float(result["score"])

    # Save to MongoDB
    record = {
        "text": text,
        "sentiment": sentiment,
        "confidence": confidence,
        "timestamp": datetime.utcnow()
    }
    collection.insert_one(record)

    return jsonify({"sentiment": sentiment, "confidence": confidence})

@app.route("/report", methods=["GET"])
def report():
    # Load records from MongoDB
    records = list(collection.find({}, {"_id": 0}))
    if not records:
        return jsonify({"error": "No data available."}), 404

    df = pd.DataFrame(records)

    # Sentiment distribution
    fig = px.pie(df, names='sentiment', title='Sentiment Distribution')
    fig.write_html("static/sentiment_pie.html")

    # Sentiment trend
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    trend = df.groupby(["date", "sentiment"]).size().reset_index(name="count")
    fig2 = px.line(trend, x="date", y="count", color="sentiment", title="Sentiment Trend Over Time")
    fig2.write_html("static/sentiment_trend.html")

    return jsonify({"message": "Reports generated successfully.",
                    "files": ["/static/sentiment_pie.html", "/static/sentiment_trend.html"]})

if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    app.run(debug=True)
