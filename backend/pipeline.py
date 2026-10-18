"""Theft Alert Console - batch pipeline.\nNeeds: pandas numpy scikit-learn joblib xgboost tensorflow (weights: config.json, lstm_autoencoder.keras, lstm_feature_scaler.joblib, xgboost_theft_model.json)
CSV (15-min readings) -> per-reading theft probability -> results.json for the dashboard.
  python pipeline.py readings.csv results.json --weights ./weights   # LSTM-AE + XGBoost (default mode)
  python pipeline.py readings.csv results.json --mode demo           # rule-based stand-in, no weights
"""
import sys, json, argparse
import numpy as np, pandas as pd

TARIFF = 11.0            # Rs per kWh, used only for the Rs estimate (edit to your tariff)
ALERT_P = 0.5            # alert threshold from the paper
SEQ = 24                 # LSTM sequence length (24 x 15 min = 6 h)
P = "Metered_Power_W"

def baseline(df):
    """Expected power = this consumer's own median for the same quarter-hour slot and day type."""
    df["slot"] = df.Timestamp.dt.hour * 4 + df.Timestamp.dt.minute // 15
    df["wkend"] = df.Timestamp.dt.dayofweek >= 5
    df["expected"] = df.groupby(["Consumer_ID", "wkend", "slot"])[P].transform("median")
    return df

def score_demo(df):
    """Transparent rule-based stand-in. NOT the trained model - uses only meter-visible columns."""
    ratio = (df[P] + 5) / (df.expected + 5)
    df["ratio"] = ratio.groupby(df.Consumer_ID).transform(lambda s: s.rolling(8, min_periods=1).mean())
    z = (0.88 - df.ratio).clip(lower=0) / 0.1 + (df.ratio - 1.2).clip(lower=0) / 0.15
    df["prob"] = 1 - np.exp(-3 * z)          # >= 0.5 once smoothed use is ~14% under or ~23% over normal
    return df

def engineer(df):
    """Features from config.json. Night = 00:00-05:59, weekend = Sat/Sun, change rate = pct change of Energy_kWh."""
    e = df.groupby("Consumer_ID").Energy_kWh.pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    df["Energy_change_rate"] = e
    df["Night_indicator"] = (df.Timestamp.dt.hour < 6).astype(int)
    df["Weekend_indicator"] = (df.Timestamp.dt.dayofweek >= 5).astype(int)
    return df

def load_model(wdir="."):
    """Load the weights once; returns score(df). df must be sorted by Consumer_ID, Timestamp with a fresh index."""
    import os; os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import joblib, xgboost as xgb, tensorflow as tf
    cfg = json.load(open(f"{wdir}/config.json"))
    F, S, XF = cfg["LSTM_FEATURE_COLS"], cfg["SEQUENCE_LENGTH"], cfg["XGB_FEATURE_COLS"]
    scaler = joblib.load(f"{wdir}/lstm_feature_scaler.joblib")
    ae = tf.keras.models.load_model(f"{wdir}/lstm_autoencoder.keras", compile=False)
    clf = xgb.XGBClassifier(); clf.load_model(f"{wdir}/xgboost_theft_model.json")

    def score(df):
        df = engineer(df)
        Xs = scaler.transform(df[F]).astype("float32")
        err = np.zeros(len(df))
        for cid, idx in df.groupby("Consumer_ID").indices.items():   # rolling window of the latest 24 readings
            X = Xs[idx]
            if len(X) < S: continue
            seqs = np.stack([X[i - S + 1:i + 1] for i in range(S - 1, len(X))])
            rec = ae.predict(seqs, batch_size=2048, verbose=0)
            err[idx[S - 1:]] = ((seqs - rec) ** 2).mean(axis=(1, 2))  # reconstruction error per reading
        df["recon_err"] = err
        Xx = np.c_[Xs, err]                                           # 10 scaled features + reconstruction error
        df["prob"] = clf.predict_proba(Xx)[:, 1]
        df.loc[err == 0, "prob"] = 0.0                                # first 23 readings have no full sequence
        contrib = clf.get_booster().predict(xgb.DMatrix(Xx, feature_names=XF), pred_contribs=True)[:, :-1]
        for k, n in enumerate(XF): df["c_" + n] = contrib[:, k]
        return df
    return score

def score_model(df, wdir="."):
    return load_model(wdir)(df)

def episodes(d):
    f = (d.prob >= ALERT_P).values
    runs, i = [], 0
    while i < len(f):
        if f[i]:
            j = i
            while j + 1 < len(f) and f[j + 1]: j += 1
            if j - i + 1 >= 4: runs.append((i, j))
            i = j + 1
        else: i += 1
    out = []
    for i, j in runs:
        s = d.iloc[i:j + 1]; r = float(((s[P] + 5) / (s.expected + 5)).mean())
        hrs = (j - i + 1) / 4
        night = s.Timestamp.dt.hour.between(0, 5).mean() > 0.6
        vi = float(((s[P] - s.Voltage_V * s.Current_A * s.Power_Factor).abs() / (s.Voltage_V * s.Current_A * s.Power_Factor + 5)).mean())
        if vi > 0.12: kind, txt = "mismatch", "Voltage and current do not match the recorded power"
        elif r < 0.12: kind, txt = "off", f"Recorded use near zero ({round((1-r)*100)}% below normal)"
        elif r < 0.9: kind, txt = "drop", f"Use {round((1-r)*100)}% below this home's normal pattern"
        elif night and r > 1.15: kind, txt = "night", "Unusual night-time use"
        elif r > 1.3: kind, txt = "spike", f"Sudden load {r:.1f}x above normal"
        else: kind, txt = "other", "Pattern deviates from this home's normal"
        out.append(dict(start=str(s.Timestamp.iloc[0]), end=str(s.Timestamp.iloc[-1]), hours=hrs, kind=kind,
                        text=txt, ratio=round(r, 2), p=round(float(s.prob.mean()), 3),
                        vmin=round(float(s.Voltage_V.min()), 1),
                        lossKwh=round(float(((s.expected - s[P]).clip(lower=0)).sum() * 0.25 / 1000), 2)))
    return out

def export(df, mode):
    cons = []
    typ_day = df.groupby(["Consumer_ID", df.Timestamp.dt.date])[P].sum().mul(0.25 / 1000).groupby("Consumer_ID").mean()
    typ = df.groupby("Consumer_ID").Consumer_Type.first()
    peer = typ_day.groupby(typ).mean()
    for cid, d in df.groupby("Consumer_ID", sort=False):
        d = d.reset_index(drop=True); fl = d.prob >= ALERT_P
        daily = fl.groupby(d.Timestamp.dt.date).mean().sort_values(ascending=False)
        risk = int(round(100 * min(1, 0.7 * daily.head(5).mean() * 2 + 0.3 * min(1, fl.mean() * 6))))
        eps = episodes(d)
        h = d.set_index("Timestamp").resample("1h")
        a, e, p = h[P].mean() / 1000, h.expected.mean() / 1000, h.prob.max()
        cc = [c for c in d.columns if c.startswith("c_")]
        drivers = []
        if cc and fl.any():
            m = d.loc[fl, cc].mean().sort_values(ascending=False)
            drivers = [dict(f=k[2:], v=round(float(v), 2)) for k, v in m.head(4).items() if v > 0]
        kinds = {}
        for x in eps: kinds[x["kind"]] = kinds.get(x["kind"], 0) + x["hours"]
        reasons = [x["text"] for x in sorted(eps, key=lambda x: -x["hours"] * x["p"])[:3]]
        cons.append(dict(id=cid, type=cons_type(d), risk=risk, flaggedPct=round(float(fl.mean()) * 100, 1),
            lossKwh=round(sum(x["lossKwh"] for x in eps), 1), episodes=eps, reasons=reasons, kinds=kinds, drivers=drivers,
            dailyKwh=round(float(typ_day[cid]), 2), peerKwh=round(float(peer[d.Consumer_Type.iloc[0]]), 2),
            t0=str(a.index[0]), actual=[round(v, 3) for v in a], expected=[round(v, 3) for v in e],
            prob=[round(float(v), 2) for v in p.fillna(0)]))
    if True:   # priority = rank among consumers (flagged share is similar across homes, so absolute scores saturate)
        order = sorted(range(len(cons)), key=lambda i: cons[i]["risk"] + 1e-3 * cons[i]["flaggedPct"])
        for rank, i in enumerate(order): cons[i]["risk"] = int(round(25 + 73 * rank / max(len(cons) - 1, 1)))
    res = dict(mode=mode, asOf=str(df.Timestamp.max()), tariff=TARIFF, threshold=ALERT_P, generated=str(pd.Timestamp.now())[:16],
               period=[str(df.Timestamp.min()), str(df.Timestamp.max())], consumers=sorted(cons, key=lambda c: -c["risk"]))
    if "Theft_Label" in df:   # simulator ground truth, for validation only
        y, yh = df.Theft_Label.values, (df.prob >= ALERT_P).values
        tp, fp, fn = int((y & yh).sum()), int((~y.astype(bool) & yh).sum()), int((y.astype(bool) & ~yh).sum())
        pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        res["validation"] = dict(precision=round(pr, 3), recall=round(rc, 3), f1=round(2 * pr * rc / max(pr + rc, 1e-9), 3), tp=tp, fp=fp, fn=fn, n=len(df))
    return res

def cons_type(d): return d.Consumer_Type.iloc[0].replace("_", " ")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("csv"); ap.add_argument("out")
    ap.add_argument("--mode", default="model", choices=["demo", "model"]); ap.add_argument("--weights", default="."); a = ap.parse_args()
    df = pd.read_csv(a.csv, parse_dates=["Timestamp"]).sort_values(["Consumer_ID", "Timestamp"]).reset_index(drop=True)
    df = baseline(df)
    df = score_demo(df) if a.mode == "demo" else score_model(df, a.weights)
    json.dump(export(df, a.mode), open(a.out, "w"), separators=(",", ":"))
    print("wrote", a.out)
