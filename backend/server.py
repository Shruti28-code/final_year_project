"""Theft Alert Console - live scoring server (deploy on Render).
Keeps the latest readings per consumer, scores every new 15-minute reading with the LSTM-AE + XGBoost
weights, and serves results to the dashboard.
  GET  /api/results   current results (same JSON the dashboard loads from a file)
  GET  /api/status    replay position / health
  POST /api/ingest    real readings: JSON list of rows (header X-API-Key if API_KEY is set)
With REPLAY=1 (default) the server replays data/readings.csv one 15-minute step every REPLAY_SECONDS
to simulate live arrival. Set REPLAY=0 when real readings are posted to /api/ingest.
"""
import os, threading, time, traceback
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
import pipeline as pl

WDIR = os.environ.get("WEIGHTS_DIR", "weights")
CSV = os.environ.get("CSV_PATH", "data/readings.csv")
WARM = int(os.environ.get("WARMUP_STEPS", 672))        # 672 x 15 min = 7 days of history before replay starts
TICK = float(os.environ.get("REPLAY_SECONDS", 10))
REPLAY = os.environ.get("REPLAY", "1") == "1"
KEY = os.environ.get("API_KEY", "")
ORIGIN = os.environ.get("CORS_ORIGIN", "*")            # set to your Vercel URL in production
SORT = ["Consumer_ID", "Timestamp"]

app = Flask(__name__)
CORS(app, origins=ORIGIN)
lock = threading.RLock()
hist = pd.DataFrame()
cache = {"res": None, "dirty": True, "t": 0.0}
state = {"ready": False, "error": None, "pos": 0, "total": 0, "done": False, "replay": REPLAY}
score = None

def ingest(new):
    """Score new readings using the last 24 stored readings of each consumer as context."""
    global hist
    with lock:
        new = new.copy(); new["Timestamp"] = pd.to_datetime(new["Timestamp"]); new["_new"] = True
        if len(hist):
            ctx = hist[[c for c in new.columns if c in hist.columns]].groupby("Consumer_ID").tail(pl.SEQ)
            ctx = ctx.assign(_new=False)
            comb = pd.concat([ctx, new], ignore_index=True)
        else:
            comb = new
        comb = comb.sort_values(SORT).reset_index(drop=True)
        out = score(comb)
        out = out[out["_new"]]
        hist = pd.concat([hist, out], ignore_index=True) if len(hist) else out
        cache["dirty"] = True

def results():
    with lock:
        if cache["res"] is None or (cache["dirty"] and time.time() - cache["t"] > 3):
            df = hist.drop(columns=["_new"], errors="ignore").sort_values(SORT).reset_index(drop=True)
            df = pl.baseline(df)
            res = pl.export(df, "model")
            res["live"] = dict(replay=state["replay"], pos=state["pos"], total=state["total"], done=state["done"])
            cache.update(res=res, dirty=False, t=time.time())
        return cache["res"]

def boot():
    global score
    try:
        score = pl.load_model(WDIR)
        df = pd.read_csv(CSV, parse_dates=["Timestamp"])
        steps = sorted(df.Timestamp.unique()); state["total"] = len(steps)
        if REPLAY:
            ingest(df[df.Timestamp.isin(steps[:WARM])]); state["pos"] = WARM
            state["ready"] = True
            while state["pos"] < len(steps):
                time.sleep(TICK)
                ingest(df[df.Timestamp == steps[state["pos"]]]); state["pos"] += 1
            state["done"] = True
        else:
            state["ready"] = True
    except Exception:
        state["error"] = traceback.format_exc(); print(state["error"], flush=True)

threading.Thread(target=boot, daemon=True).start()

@app.get("/")
def root(): return jsonify(ok=True, service="theft-alert-console", ready=state["ready"])

@app.get("/api/status")
def status(): return jsonify(state)

@app.get("/api/results")
def api_results():
    if state["error"]: return jsonify(error="startup failed", detail=state["error"][-800:]), 500
    if not state["ready"] or hist.empty: return jsonify(loading=True), 503
    return jsonify(results())

@app.post("/api/ingest")
def api_ingest():
    if KEY and request.headers.get("X-API-Key") != KEY: return jsonify(error="unauthorized"), 401
    if not state["ready"]: return jsonify(loading=True), 503
    rows = pd.DataFrame(request.get_json(force=True))
    need = ["Timestamp", "Consumer_ID", "Consumer_Type", "Voltage_V", "Current_A", "Apparent_Power_VA", "Reactive_Power_VAR",
            "Frequency_Hz", "Energy_kWh", "Power_Factor", "Metered_Power_W"]
    miss = [c for c in need if c not in rows.columns]
    if miss: return jsonify(error="missing columns", columns=miss), 400
    ingest(rows); return jsonify(ok=True, n=len(rows))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
