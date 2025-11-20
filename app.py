import os
from datetime import datetime, timedelta

import numpy as np
import yfinance as yf
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras.models import load_model

# ----- Load Model + Scaler -----
MODEL_PATH = "lstm_model.h5"
SCALER_PATH = "scaler.pkl"
LOOKBACK_PATH = "lookback.txt"

if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(LOOKBACK_PATH)):
    raise RuntimeError("Files missing. Ensure model, scaler, and lookback.txt are in the same folder.")

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
with open(LOOKBACK_PATH, "r") as f:
    LOOKBACK = int(f.read().strip())

# ----- FastAPI App -----
app = FastAPI(title="Stock LSTM Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    ticker: str
    days: int = 7

import requests
import yfinance as yf

def download_close_prices(ticker: str, lookback: int):
    """
    Download daily close prices for the last 5 years using yfinance.
    Uses period-based download to ensure maximum compatibility.
    """

    # Force yfinance to use a browser-like user agent
    yf.utils.default_user_agent = lambda: "Mozilla/5.0"

    try:
        df = yf.download(
            tickers=ticker,
            period="5y",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception as e:
        raise ValueError(f"Yahoo Finance request failed: {e}")

    if df is None or df.empty:
        raise ValueError(f"No data available for '{ticker}'. Try another symbol.")

    if "Close" not in df.columns:
        raise ValueError(f"No 'Close' column returned for '{ticker}'.")

    close = df["Close"].dropna().values.reshape(-1, 1)

    if len(close) < lookback:
        raise ValueError(f"Not enough data for '{ticker}'. Needed {lookback}, got {len(close)}.")

    return df.index, close

def make_future_predictions(close, days):
    scaled = scaler.transform(close)
    current = scaled[-LOOKBACK:].copy()
    preds_scaled = []

    for _ in range(days):
        inp = current.reshape(1, LOOKBACK, 1)
        next_scaled = model.predict(inp, verbose=0)[0, 0]
        preds_scaled.append(next_scaled)
        current = np.append(current[1:], [[next_scaled]], axis=0)

    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    preds = scaler.inverse_transform(preds_scaled)
    return preds.flatten()

@app.get("/")
def root():
    return {"message": "Stock LSTM API is running!"}

@app.post("/predict")
def predict(req: PredictRequest):
    try:
        dates, close = download_close_prices(req.ticker, LOOKBACK)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    future_prices = make_future_predictions(close, req.days)
    last_date = dates[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, req.days + 1)]

    return {
        "ticker": req.ticker.upper(),
        "predictions": [
            {"date": future_dates[i].strftime("%Y-%m-%d"), "price": float(future_prices[i])}
            for i in range(req.days)
        ],
    }
