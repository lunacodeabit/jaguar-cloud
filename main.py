import asyncio
import json
import math
import os
import pandas as pd
import numpy as np
import ccxt.async_support as ccxt
from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn

# --- CONFIGURATION ---
# We use Kraken to avoid US blocking issues on Render
EXCHANGE_ID = 'kraken'
SYMBOL = 'BTC/USD'
LIMIT = 250

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory=".")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# --- MATH ENGINE ---
def calculate_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def calculate_mfi(high, low, close, volume, period=14):
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume
    positive_flow = []
    negative_flow = []
    for i in range(1, len(typical_price)):
        if typical_price.iloc[i] > typical_price.iloc[i-1]:
            positive_flow.append(raw_money_flow.iloc[i])
            negative_flow.append(0)
        elif typical_price.iloc[i] < typical_price.iloc[i-1]:
            positive_flow.append(0)
            negative_flow.append(raw_money_flow.iloc[i])
        else:
            positive_flow.append(0)
            negative_flow.append(0)
    positive_flow = [0] + positive_flow
    negative_flow = [0] + negative_flow
    positive_mf = pd.Series(positive_flow, index=typical_price.index)
    negative_mf = pd.Series(negative_flow, index=typical_price.index)
    mfr = positive_mf.rolling(window=period).sum() / negative_mf.rolling(window=period).sum()
    mfi = 100 - (100 / (1 + mfr))
    return mfi

def calculate_hma(series, period):
    def wma(s, p):
        return s.rolling(p).apply(lambda x: ((x * np.arange(1, p + 1)).sum()) / np.arange(1, p + 1).sum(), raw=True)
    wma_half = wma(series, int(period / 2))
    wma_full = wma(series, period)
    raw_hma = 2 * wma_half - wma_full
    hma_sqrt = wma(raw_hma, int(math.sqrt(period)))
    return hma_sqrt

# --- JAGUAR INTELLIGENCE ---
async def get_market_data(timeframe):
    exchange = ccxt.kraken()
    try:
        # Fetch OHLCV data from Kraken
        ohlcv = await exchange.fetch_ohlcv(SYMBOL, timeframe, limit=LIMIT)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Calculate Indicators
        df['hma_90'] = calculate_hma(df['close'], 90)
        df['ema_200'] = calculate_ema(df['close'], 200)
        df['mfi'] = calculate_mfi(df['high'], df['low'], df['close'], df['volume'], 14)
        
        df.dropna(inplace=True)
        if len(df) < 2: return None
        last = df.iloc[-1]
        
        # Scoring Logic
        score = 5.0
        trend = "NEUTRAL"
        
        # Trend Filter (EMA 200)
        if last['close'] > last['ema_200']:
            trend = "BULLISH"
            score += 2.0
        else:
            trend = "BEARISH"
            score -= 2.0
            
        # Momentum Filter (HMA 90)
        if last['close'] > last['hma_90']: score += 1.5
        else: score -= 1.5
        
        # Fuel Filter (MFI)
        if last['mfi'] > 80: score -= 1.0 # Overbought
        elif last['mfi'] < 20: score += 1.0 # Oversold
        
        score = max(0.0, min(10.0, score))

        # Signal Logic for Chart Markers
        signal = None
        if score >= 7.5: signal = "LONG"
        elif score <= 2.5: signal = "SHORT"

        payload = {
            "timeframe": timeframe,
            "price": float(last['close']),
            "trend": trend,
            "score": round(score, 1),
            "mfi": int(last['mfi']),
            "signal": signal,
            "candle": {
                "time": int(last['timestamp'] / 1000),
                "open": float(last['open']),
                "high": float(last['high']),
                "low": float(last['low']),
                "close": float(last['close'])
            }
        }
        return payload
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None
    finally:
        await exchange.close()

# --- WEBSOCKET MANAGER ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    current_timeframe = '1m' # Default timeframe
    
    # Task 1: Listen for timeframe changes from the user
    async def listen_for_commands():
        nonlocal current_timeframe
        try:
            async for message in websocket.iter_text():
                data = json.loads(message)
                if 'timeframe' in data:
                    current_timeframe = data['timeframe']
                    print(f"Timeframe switched to: {current_timeframe}")
        except WebSocketDisconnect:
            print("Client disconnected (Listener)")

    # Task 2: Push market data constantly
    async def send_data_loop():
        try:
            while True:
                data = await get_market_data(current_timeframe)
                if data:
                    await websocket.send_text(json.dumps(data))
                await asyncio.sleep(2) # Kraken rate limit safety
        except Exception as e:
            print(f"Error in sender loop: {e}")

    # Run both tasks simultaneously
    await asyncio.gather(listen_for_commands(), send_data_loop())

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
