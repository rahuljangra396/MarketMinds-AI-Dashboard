"""
MarketMinds - AI Stock & Crypto Analysis Dashboard
By Rahi

Features:
✅ Live Stock/Crypto prices & charts (yfinance)
✅ SMA/EMA indicators
✅ Basic AI-based next-day prediction
✅ Portfolio tracker (save locally)
✅ Financial News + Sentiment Analysis
✅ AI Insight with OpenAI (optional)
✅ Alerts for price drops
✅ Clean Streamlit UI

Requires: streamlit, yfinance, pandas, numpy, plotly, sklearn, vaderSentiment, openai, requests, python-dotenv
"""

import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LinearRegression
from datetime import datetime
import requests
from dotenv import load_dotenv
from openai import OpenAI

# ---------------- SETUP ----------------
load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
NEWS_KEY = os.getenv("NEWSAPI_KEY", "")

client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None

PORTFOLIO_FILE = "portfolio.json"

st.set_page_config(page_title="MarketMinds", layout="wide", page_icon="📊")

# ---------------- SIDEBAR ----------------
st.sidebar.title("📊 MarketMinds Dashboard")
st.sidebar.write("An AI-powered financial analytics app")

ticker = st.sidebar.text_input("Enter Stock/Crypto Symbol (e.g. AAPL, TSLA, BTC-USD):", "AAPL").upper()
period = st.sidebar.selectbox("Select Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=2)
interval = st.sidebar.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)

st.sidebar.markdown("---")
st.sidebar.header("📁 Portfolio Manager")

if os.path.exists(PORTFOLIO_FILE):
    with open(PORTFOLIO_FILE, "r") as f:
        portfolio = json.load(f)
else:
    portfolio = {}

new_symbol = st.sidebar.text_input("Add symbol to portfolio:")
new_qty = st.sidebar.number_input("Quantity", min_value=0.0, step=1.0)

if st.sidebar.button("Add"):
    if new_symbol:
        portfolio[new_symbol.upper()] = portfolio.get(new_symbol.upper(), 0) + new_qty
        with open(PORTFOLIO_FILE, "w") as f:
            json.dump(portfolio, f, indent=2)
        st.sidebar.success(f"Added {new_qty} of {new_symbol.upper()}!")

if st.sidebar.button("Clear Portfolio"):
    portfolio = {}
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(portfolio, f)
    st.sidebar.info("Portfolio cleared.")

st.sidebar.write("### Current Portfolio:")
if portfolio:
    st.sidebar.json(portfolio)
else:
    st.sidebar.write("No holdings yet.")

# ---------------- MAIN SECTION ----------------
st.title("💹 MarketMinds – AI Stock & Crypto Analysis")
st.markdown("Real-time data, AI insights, and trend predictions for smarter investing.")

try:
    data = yf.Ticker(ticker).history(period=period, interval=interval)
    if data.empty:
        st.error("No data found. Please try another symbol.")
        st.stop()
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

col1, col2 = st.columns(2)
latest_close = data["Close"].iloc[-1]
prev_close = data["Close"].iloc[-2]
price_change = latest_close - prev_close
col1.metric("Current Price", f"${latest_close:.2f}", f"{price_change:+.2f}")

# Plot price chart
data["SMA_20"] = data["Close"].rolling(window=20).mean()
data["SMA_50"] = data["Close"].rolling(window=50).mean()

fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data["Close"], mode="lines", name="Close"))
fig.add_trace(go.Scatter(x=data.index, y=data["SMA_20"], mode="lines", name="SMA 20"))
fig.add_trace(go.Scatter(x=data.index, y=data["SMA_50"], mode="lines", name="SMA 50"))
fig.update_layout(title=f"{ticker} Price Chart", height=450, xaxis_title="Date", yaxis_title="Price ($)")
st.plotly_chart(fig, use_container_width=True)

# ---------------- PREDICTION ----------------
st.subheader("📈 AI-Based Price Prediction (Simple Linear Regression)")
X = np.arange(len(data)).reshape(-1, 1)
y = data["Close"].values
model = LinearRegression().fit(X, y)
pred = model.predict([[len(data) + 1]])[0]
st.write(f"Predicted next close: **${pred:.2f}**")

# ---------------- PORTFOLIO VALUE ----------------
st.subheader("📊 Portfolio Overview")

if portfolio:
    table = []
    total_value = 0.0

    # Safely handle stock portfolio values
    for s in portfolio.get("stocks", []):
        if isinstance(s, dict):
            sym = s.get("symbol", "N/A")
            qty = float(s.get("quantity", 0))
            price = float(s.get("price", 0))
            value = qty * price
            total_value += value
            table.append([sym, qty, price, value])

    # Safely handle crypto portfolio values
    for c in portfolio.get("crypto", []):
        if isinstance(c, dict):
            sym = c.get("symbol", "N/A")
            qty = float(c.get("quantity", 0))
            price = float(c.get("price", 0))
            value = qty * price
            total_value += value
            table.append([sym, qty, price, value])

    if table:
        df = pd.DataFrame(table, columns=["Symbol", "Qty", "Price", "Value"])
        st.table(df)
        st.success(f"💰 Total Portfolio Value: ${total_value:,.2f}")
    else:
        st.info("No portfolio data yet.")
else:
    st.info("No portfolio data yet.")

# Safely handle portfolio values
for s in portfolio.get("stocks", []):
    if isinstance(s, (int, float)):
        total_value += s

for c in portfolio.get("crypto", []):
    if isinstance(c, (int, float)):
        total_value += c

        table.append([sym, qty, price, value])
    df = pd.DataFrame(table, columns=["Symbol", "Qty", "Price", "Value"])
    st.table(df)
    st.success(f"Total Portfolio Value: ${total_value:.2f}")
else:
    st.info("No portfolio data yet.")

# ---------------- NEWS & SENTIMENT ----------------
st.subheader("📰 News & Sentiment Analysis")
if NEWS_KEY:
    url = f"https://newsapi.org/v2/everything?q={ticker}&pageSize=5&apiKey={NEWS_KEY}"
    try:
        res = requests.get(url).json()
        articles = res.get("articles", [])
        if not articles:
            st.warning("No news found.")
        else:
            analyzer = SentimentIntensityAnalyzer()
            sentiments = []
            for a in articles:
                title = a["title"]
                score = analyzer.polarity_scores(title)["compound"]
                sentiments.append(score)
                st.write(f"**{title}**")
                st.progress((score + 1) / 2)
            avg_sent = np.mean(sentiments)
            st.metric("Average Sentiment Score", f"{avg_sent:.3f}")
    except Exception as e:
        st.error(f"Error fetching news: {e}")
else:
    st.info("Add your NewsAPI key in .env to see headlines.")

# ---------------- AI INSIGHT ----------------
st.subheader("🧠 AI Market Insight")
if st.button("Generate Insight"):
    if client:
        try:
            closes = list(data["Close"].tail(10).round(2))
            prompt = f"Given these 10 closing prices of {ticker}: {closes}, give a brief AI-generated insight about the market sentiment and a 1-sentence investment tip."
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=120,
                temperature=0.6,
            )
            st.info(response.choices[0].message.content.strip())
        except Exception as e:
            error_str = str(e)
            if "insufficient_quota" in error_str or "429" in error_str:
                st.warning("OpenAI quota exceeded. Using basic insight instead.")
                # Fall back to basic insight
                sma_20 = data["SMA_20"].iloc[-1]
                if pd.isna(sma_20):
                    insight = f"Insufficient data for {ticker} to generate insight."
                else:
                    if latest_close > sma_20:
                        sentiment = "bullish"
                        tip = "Consider holding or buying if it aligns with your strategy."
                    else:
                        sentiment = "bearish"
                        tip = "Consider selling or waiting for better conditions."
                    insight = f"The market sentiment for {ticker} appears {sentiment} based on current price (${latest_close:.2f}) vs. SMA_20 (${sma_20:.2f}). {tip}"
                st.info(insight)
            else:
                st.error(f"OpenAI Error: {e}")
    else:
        # Basic insight based on technical indicators
        sma_20 = data["SMA_20"].iloc[-1]
        if pd.isna(sma_20):
            insight = f"Insufficient data for {ticker} to generate insight."
        else:
            if latest_close > sma_20:
                sentiment = "bullish"
                tip = "Consider holding or buying if it aligns with your strategy."
            else:
                sentiment = "bearish"
                tip = "Consider selling or waiting for better conditions."
            insight = f"The market sentiment for {ticker} appears {sentiment} based on current price (${latest_close:.2f}) vs. SMA_20 (${sma_20:.2f}). {tip}"
        st.info(insight)

# ---------------- ALERT ----------------
st.subheader("🚨 Price Alert")
threshold = st.number_input("Alert me if price falls below:", value=latest_close * 0.95)
if latest_close <= threshold:
    st.error(f"⚠️ Alert: {ticker} has dropped below ${threshold:.2f}!")
else:
    st.success(f"✅ Price above alert level (${threshold:.2f})")

st.markdown("---")
st.caption("© 2025 MarketMinds | Created by Rahi | Powered by Streamlit + OpenAI + Yahoo Finance")
