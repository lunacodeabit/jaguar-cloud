import asyncio
import json
import math
import os
import pandas as pd
import numpy as np
import ccxt.async_support as ccxt
from fastapi import FastAPI, WebSocket, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn

# --- CONFIGURACIÓN ---
# CAMBIO: Usamos KRAKEN porque funciona en servidores de USA (Render)
EXCHANGE_ID = 'kraken'
SYMBOL = 'BTC/USD' 
TIMEFRAME = '1m'
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

# --- MATEMÁTICAS ---
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

# --- EL CEREBRO ---
async def get_market_data():
    # Usamos Kraken para evitar bloqueo de USA
    exchange = ccxt.kraken() 
    try:
        ohlcv = await exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=LIMIT)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        df['hma_90'] = calculate_hma(df['close'], 90)
        df['ema_200'] = calculate_ema(df['close'], 200)
        df['mfi'] = calculate_mfi(df['high'], df['low'], df['close'], df['volume'], 14)
        
        df.dropna(inplace=True)
        if len(df) < 2: return None
        last = df.iloc[-1]
        
        score = 5.0
        trend = "NEUTRAL"
        if last['close'] > last['ema_200']:
            trend = "ALCISTA"
            score += 2.0
        else:
            trend = "BAJISTA"
            score -= 2.0
        if last['close'] > last['hma_90']: score += 1.5
        else: score -= 1.5
        if last['mfi'] > 80: score -= 1.0
        elif last['mfi'] < 20: score += 1.0
        score = max(0.0, min(10.0, score))

        payload = {
            "price": float(last['close']),
            "trend": trend,
            "score": round(score, 1),
            "mfi": int(last['mfi']),
            "signal": "LONG" if score > 7.5 else "WAIT",
            # Datos extra para el gráfico de velas
            "candle": {
                "time": int(last['timestamp'] / 1000), # TradingView usa segundos
                "open": float(last['open']),
                "high": float(last['high']),
                "low": float(last['low']),
                "close": float(last['close'])
            }
        }
        return payload
    except Exception as e:
        print(f"Error: {e}")
        return None
    finally:
        await exchange.close()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await get_market_data()
            if data:
                await websocket.send_text(json.dumps(data))
            await asyncio.sleep(2) # Kraken es más estricto con rate limits, subimos a 2s
    except Exception as e:
        print("Desconectado")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)