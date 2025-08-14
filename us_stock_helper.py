# -*- coding: utf-8 -*-
"""US-stock-helper (Revised: leakage fix, proper backtest, business days, CI-friendly)"""

import warnings
warnings.filterwarnings("ignore")

import io, requests
import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date, timedelta
from tqdm import tqdm

# Prophet baseline
from prophet import Prophet

# Regression
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Kalman local linear trend & ARIMA
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.arima.model import ARIMA

# Macro
from fredapi import Fred

# ========= Config =========
# 建议将真实 key 放到环境变量中
FRED_API_KEY = os.getenv("FRED_API_KEY")
fred = Fred(api_key=FRED_API_KEY)

TICKERS = ["LLY", "NVO", "AAPL", "META", "NVDA", "AMZN", "TSLA", "AMD", "MSFT", "GOOGL"]
SECTOR_ETFS = ["XLF", "XLK", "XLE", "XLV"]
START_DATE = "2015-01-01"
END_DATE = date.today().strftime("%Y-%m-%d")

BASE_FRED_IDS = ["DGS10","DGS2","T10Y2Y","CPIAUCSL","UNRATE","VIXCLS","DTWEXB","DCOILWTICO"]
ADDITIONAL_FRED_IDS = ["USSLIND","UMCSENT"]
GOLD_CANDIDATES = ["IR14270","GOLDPMGBD228NLBM","GOLDAMGBD228NLBM"]

# ========= Utils =========
STD_FIELDS = {"Open","High","Low","Close","Adj Close","Volume"}

def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    try:
        df.index = df.index.tz_localize(None)
    except Exception:
        pass
    return df.sort_index()

def _force_single_level_ohlcv(df: pd.DataFrame, ticker: str|None=None) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    last = list(df.columns.get_level_values(-1))
    if len(set(last) & STD_FIELDS) >= 3:
        df.columns = last
        return df
    if ticker is not None and ticker in df.columns.get_level_values(-1):
        df = df.xs(ticker, axis=1, level=-1)
        if isinstance(df.columns, pd.MultiIndex):
            last = list(df.columns.get_level_values(-1))
            if len(set(last) & STD_FIELDS) >= 3:
                df.columns = last
            else:
                df.columns = ["_".join([str(x) for x in tup if str(x)]) for tup in df.columns]
        return df
    df.columns = ["_".join([str(x) for x in tup if str(x)]) for tup in df.columns]
    return df

# ========= Data download (adj prices) =========
def download_adj_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, threads=False, progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise ValueError(f"{ticker} 无法下载价格数据")
    df = _ensure_datetime_index(_force_single_level_ohlcv(df, ticker))
    if "Close" not in df.columns:
        if "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]
        else:
            raise ValueError(f"{ticker} 数据缺少 Close 列，实际列: {list(df.columns)}")
    for c in ["Close","Open","High","Low","Adj Close","Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def download_etf_adj_close(ticker: str, start: str, end: str) -> pd.Series:
    df = download_adj_ohlcv(ticker, start, end)
    s = df["Close"].copy()
    s.name = ticker
    return s

# ========= Technicals =========
def compute_technicals_from_close(close: pd.Series) -> pd.DataFrame:
    px = pd.DataFrame(index=close.index.copy())
    px["Close"] = close.astype(float)
    px["Return"] = px["Close"].pct_change()
    px["Volatility20"] = px["Return"].rolling(20).std()
    px["MA20"] = px["Close"].rolling(20).mean()
    px["MA50"] = px["Close"].rolling(50).mean()
    px["MA200"] = px["Close"].rolling(200).mean()
    px["ROC20"] = px["Close"].pct_change(20)
    delta = px["Close"].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    rs = up / (down.replace(0, np.nan))
    px["RSI14"] = 100 - (100 / (1 + rs))
    ema12 = px["Close"].ewm(span=12, adjust=False).mean()
    ema26 = px["Close"].ewm(span=26, adjust=False).mean()
    px["MACD"] = ema12 - ema26
    px["MACDsig"] = px["MACD"].ewm(span=9, adjust=False).mean()
    px["MACDhist"] = px["MACD"] - px["MACDsig"]
    return px

def build_stock_features(ticker: str, start: str, end: str) -> pd.DataFrame:
    px = download_adj_ohlcv(ticker, start, end)
    tech = compute_technicals_from_close(px["Close"])
    px = px.join(tech.drop(columns=["Close"], errors="ignore"))
    return px

# ========= FRED fetch =========
def get_fred_series_csv(series_id: str, start: str):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        r = requests.get(url, timeout=20, headers={'User-Agent':'Mozilla/5.0'})
        if r.status_code != 200 or 'text/html' in r.headers.get('Content-Type',''):
            return None
        df = pd.read_csv(io.StringIO(r.text))
        date_col = [c for c in df.columns if c.upper().startswith("DATE")]
        if date_col:
            df[date_col[0]] = pd.to_datetime(df[date_col[0]])
            df = df.set_index(date_col[0])
        s = df.iloc[:, 1 if df.shape[1] > 1 else 0].replace(".", np.nan).astype(float)
        s.name = series_id
        s = s[s.index >= pd.to_datetime(start)]
        s.index = s.index.tz_localize(None)
        return s.sort_index()
    except Exception:
        return None

def get_fred_series_robust(series_id: str, start: str) -> pd.Series:
    try:
        s = fred.get_series(series_id, observation_start=start)
        if s is not None and len(s) > 0:
            s = s.rename(series_id)
            s.index = pd.to_datetime(s.index).tz_localize(None)
            return s.sort_index()
    except Exception:
        pass
    s = get_fred_series_csv(series_id, start)
    return s if s is not None else pd.Series(name=series_id, dtype=float)

def first_available_series(candidates, start: str):
    for sid in candidates:
        s = get_fred_series_robust(sid, start)
        if s is not None and not s.empty:
            return sid, s
    return None, None

def build_macro_frame(start: str) -> pd.DataFrame:
    frames = []
    for sid in BASE_FRED_IDS + ADDITIONAL_FRED_IDS:
        s = get_fred_series_robust(sid, start)
        if s is not None and not s.empty:
            frames.append(s)
    df = pd.concat(frames, axis=1) if frames else pd.DataFrame()

    gold_id, gold_s = first_available_series(GOLD_CANDIDATES, start)
    if gold_s is not None:
        gold_s.name = "GOLD_USD"
        df = gold_s.to_frame() if df.empty else df.join(gold_s, how="outer")

    if not df.empty:
        df = _ensure_datetime_index(df).resample("D").ffill().bfill()

    if "T10Y2Y" not in df.columns and all(c in df.columns for c in ["DGS10","DGS2"]):
        df["T10Y2Y"] = df["DGS10"] - df["DGS2"]

    return df

def get_sector_etf_data(etfs, start: str, end: str) -> pd.DataFrame:
    frames = []
    for etf in etfs:
        try:
            s = download_etf_adj_close(etf, start, end)
            frames.append(s)
        except Exception as e:
            print(f"[WARN] 板块ETF {etf} 获取失败: {e}")
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1)
    out = _ensure_datetime_index(out).resample("D").ffill().bfill()
    return out

# ========= Econ cycle =========
def classify_economic_cycle(df: pd.DataFrame) -> pd.Series:
    if "USSLIND" not in df.columns or "UMCSENT" not in df.columns:
        return pd.Series(index=df.index, data="Unknown", name="EconCycle", dtype="category")
    li = df["USSLIND"]; cs = df["UMCSENT"]
    conds = [
        (li > 0) & (cs > 80),
        (li > 0) & (cs <= 80),
        (li <= 0) & (cs <= 80),
        (li <= 0) & (cs > 80),
    ]
    choices = ["Boom","Slowdown","Recession","Recovery"]
    res = np.select(conds, choices, default="Unknown")
    return pd.Series(res, index=df.index, name="EconCycle", dtype="category")

# ========= Prophet (price, calibrated) =========
def prophet_forecast(close: pd.Series, horizon_days: int = 180) -> pd.DataFrame:
    close = close.dropna().astype(float)
    if len(close) < 10:
        last_price = float(close.iloc[-1])
        future_dates = pd.bdate_range(close.index.max() + timedelta(days=1), periods=horizon_days)
        return pd.DataFrame({"ds": future_dates, "yhat": last_price, "yhat_lower": last_price, "yhat_upper": last_price})
    df_prophet = pd.DataFrame({"ds": close.index, "y": close.values})
    m = Prophet()
    m.fit(df_prophet)
    future = m.make_future_dataframe(periods=horizon_days, freq="B")  # 交易日
    fc = m.predict(future)

    # 校准到最后一个真实价格
    last_actual = float(close.iloc[-1])
    # 找到与最后交易日最接近的 fc 行
    idx_last = (fc["ds"] - close.index.max()).abs().idxmin()
    last_pred = float(fc.loc[idx_last, "yhat"])
    if np.isfinite(last_pred) and last_pred != 0:
        scale = last_actual / last_pred
        for c in ["yhat","yhat_lower","yhat_upper"]:
            fc[c] = fc[c] * scale

    return fc[["ds","yhat","yhat_lower","yhat_upper"]]

# ========= Kalman local linear trend =========
def kalman_trend_and_forecast(close: pd.Series, horizon_days: int = 180):
    close = close.dropna().astype(float)
    if len(close) < 5:
        future_dates = pd.bdate_range(close.index.max() + timedelta(days=1), periods=horizon_days)
        last_price = float(close.iloc[-1])
        dummy = pd.DataFrame({"ds": future_dates, "yhat": last_price, "yhat_lower": last_price, "yhat_upper": last_price})
        return pd.Series(index=close.index, data=last_price, name="KalmanTrend"), close, dummy

    mod = UnobservedComponents(close, level="local linear trend")
    res = mod.fit(disp=False)

    # Robust retrieval of smoothed level
    st = None
    if hasattr(res, "smoothed_state"):
        try: st = res.smoothed_state
        except Exception: st = None
    if st is None and hasattr(res, "states"):
        try:
            if hasattr(res.states, "smoothed"):
                st = res.states.smoothed
        except Exception: st = None
    if st is None and hasattr(res, "level_smoothed"):
        try: st = res.level_smoothed
        except Exception: st = None

    level = None
    try:
        if st is None:
            level = pd.Series(index=close.index, data=np.nan)
        else:
            if isinstance(st, np.ndarray):
                level = pd.Series(st[0], index=close.index)
            elif isinstance(st, (pd.DataFrame, pd.Series)):
                try:
                    level_candidate = st.iloc[0]
                    if isinstance(level_candidate, pd.Series):
                        level = pd.Series(level_candidate.values, index=close.index)
                    else:
                        level = pd.Series(np.asarray(level_candidate), index=close.index)
                except Exception:
                    arr = np.asarray(st)
                    level = pd.Series(arr[0], index=close.index)
            else:
                arr = np.asarray(st)
                level = pd.Series(arr[0], index=close.index)
    except Exception:
        level = pd.Series(index=close.index, data=np.nan)

    level.name = "KalmanTrend"

    # 一步预测历史拟合
    kalman_hist = res.fittedvalues.reindex(close.index)

    # 预测未来
    try:
        fc = res.get_forecast(steps=horizon_days)
        mean = fc.predicted_mean
        ci = fc.conf_int()
        lower = ci.iloc[:, 0].values
        upper = ci.iloc[:, 1].values
        out = pd.DataFrame({
            "ds": pd.bdate_range(close.index.max() + timedelta(days=1), periods=horizon_days),
            "yhat": np.array(mean, dtype=float),
            "yhat_lower": np.array(lower, dtype=float),
            "yhat_upper": np.array(upper, dtype=float)
        })
    except Exception:
        last_price = float(close.iloc[-1])
        future_dates = pd.bdate_range(close.index.max() + timedelta(days=1), periods=horizon_days)
        out = pd.DataFrame({"ds": future_dates, "yhat": last_price, "yhat_lower": last_price, "yhat_upper": last_price})

    return level, kalman_hist, out

# ========= ARIMA on returns (with hist one-step) =========
def arima_returns_forecast(close: pd.Series, horizon_days: int = 180):
    close = close.dropna().astype(float)
    if len(close) < 40:
        last_price = float(close.iloc[-1])
        future_dates = pd.bdate_range(close.index.max() + timedelta(days=1), periods=horizon_days)
        dummy_fc = pd.DataFrame({"ds": future_dates,
                                 "yhat": last_price,
                                 "yhat_lower": last_price,
                                 "yhat_upper": last_price})
        # 历史一步预测（不足时用朴素）
        hist = close.shift(1).reindex(close.index)
        return dummy_fc, hist

    try:
        logp = np.log(close)
        ret = logp.diff().dropna()  # index: t (从第二个开始)

        model = ARIMA(ret, order=(1,0,1))
        res = model.fit()

        # 历史一步预测的“收益”
        pred_hist = res.get_prediction(dynamic=False)
        mean_ret_hist = pd.Series(pred_hist.predicted_mean, index=ret.index)

        # 历史一步预测 -> 价格（预测 P_t = P_{t-1} * exp(ret_t_pred)）
        arima_hist = (close.shift(1).reindex(mean_ret_hist.index) * np.exp(mean_ret_hist)).dropna()

        # 未来收益预测
        fc = res.get_forecast(steps=horizon_days)
        mean_ret = fc.predicted_mean.values
        ci_ret = fc.conf_int()
        lower_ret = ci_ret.iloc[:,0].values
        upper_ret = ci_ret.iloc[:,1].values

        last_price = float(close.iloc[-1])
        yhat = last_price * np.exp(np.cumsum(mean_ret))
        yhat_lower = last_price * np.exp(np.cumsum(lower_ret))
        yhat_upper = last_price * np.exp(np.cumsum(upper_ret))
        out = pd.DataFrame({
            "ds": pd.bdate_range(close.index.max() + timedelta(days=1), periods=horizon_days),
            "yhat": yhat,
            "yhat_lower": yhat_lower,
            "yhat_upper": yhat_upper
        })
        return out, arima_hist
    except Exception:
        last_price = float(close.iloc[-1])
        future_dates = pd.bdate_range(close.index.max() + timedelta(days=1), periods=horizon_days)
        dummy_fc = pd.DataFrame({"ds": future_dates,
                                 "yhat": last_price,
                                 "yhat_lower": last_price,
                                 "yhat_upper": last_price})
        hist = close.shift(1).reindex(close.index)
        return dummy_fc, hist

# ========= Lasso factor forecast (lag features, next-day returns) =========
def lasso_factor_forecast_rolling(features: pd.DataFrame, horizon_days: int = 180):
    df = features.copy()
    # 数值清洗
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan).ffill().bfill()

    # 目标：下一期对数收益
    df["logret"] = np.log(df["Close"]).diff()
    y = df["logret"].shift(-1)

    # 特征：去掉目标与 Close，并整体滞后一天
    X = df.drop(columns=["logret", "Close"], errors="ignore").select_dtypes(include=[np.number]).shift(1)

    # 对齐有效样本
    valid = y.dropna().index.intersection(X.dropna().index)
    X = X.loc[valid]
    y = y.loc[valid]

    last_price = float(df["Close"].iloc[-1])
    future_dates = pd.bdate_range(df.index.max() + timedelta(days=1), periods=horizon_days)

    if len(y) < 120 or X.shape[1] < 1:
        # 未来预测
        fc = pd.DataFrame({"ds": future_dates, "yhat": last_price, "yhat_lower": last_price, "yhat_upper": last_price})
        # 历史一步预测（朴素兜底）
        hist = df["Close"].shift(1).reindex(df.index)
        return fc, hist

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("lasso", LassoCV(cv=5, random_state=0))
    ])
    pipe.fit(X, y)

    # 历史一步预测：pred_ret_t 对应 t+1 的收益；为与 close 对齐到“当日 t”的价格，计算
    # P_t_hat = P_{t-1} * exp(pred_ret_{t-1}) -> 直接用 shift(1) 对齐
    pred_ret_hist = pd.Series(pipe.predict(X), index=X.index)
    lasso_hist = (df["Close"].shift(1).reindex(pred_ret_hist.index) * np.exp(pred_ret_hist)).dropna()

    # 残差波动用于简单置信区间
    resid = y - pipe.predict(X)
    sigma = float(np.nanstd(resid))
    sigma = sigma if np.isfinite(sigma) and sigma > 0 else 0.0

    # 逐步模拟未来路径：用“上一期特征”预测“下一期收益”
    close_hist = df["Close"].astype(float).copy()
    close_sim = close_hist.copy()
    preds, lower, upper = [], [], []
    price = last_price

    # 以训练的列名为准构造未来单行特征
    feat_cols = X.columns.tolist()
    last_num_row = df.iloc[-1][df.select_dtypes(include=[np.number]).columns]

    for step in range(horizon_days):
        tech = compute_technicals_from_close(close_sim)

        row = last_num_row.copy()
        for col in ["Return","Volatility20","MA20","MA50","MA200","ROC20","RSI14","MACD","MACDsig","MACDhist",
                    "Open","High","Low","Adj Close","Volume"]:
            if col in tech.columns:
                row[col] = tech.iloc[-1].get(col, row.get(col, np.nan))

        # 与训练列对齐（训练时已整体 shift(1)）
        row = row.reindex(feat_cols)
        row = row.replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(method="bfill").fillna(0.0)

        try:
            pred_logret = float(pipe.predict(row.values.reshape(1, -1))[0])
        except Exception:
            pred_logret = 0.0

        price = price * float(np.exp(pred_logret))
        close_sim.loc[close_sim.index.max() + timedelta(days=1)] = price

        preds.append(price)
        if sigma > 0:
            # 近似独立同分布：方差随步数累积
            cum = float(np.sqrt(step + 1))
            lower.append(last_price * np.exp(-1.96 * sigma * cum))
            upper.append(last_price * np.exp(+1.96 * sigma * cum))
        else:
            lower.append(price); upper.append(price)

    fc = pd.DataFrame({"ds": future_dates, "yhat": preds, "yhat_lower": lower, "yhat_upper": upper})
    return fc, lasso_hist

def compute_feature_importance(features_df: pd.DataFrame) -> pd.DataFrame:
    df = features_df.copy()
    # 与训练定义一致：下一期收益 + 滞后特征
    df["logret"] = np.log(df["Close"]).diff()
    y = df["logret"].shift(-1)

    X = df.drop(columns=["logret", "Close"], errors="ignore").select_dtypes(include=[np.number]).shift(1)

    # 清洗 & 对齐
    X = X.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    valid = y.dropna().index.intersection(X.dropna().index)
    X = X.loc[valid]
    y = y.loc[valid]

    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("清洗后样本数为 0，跳过 Lasso 重要性")

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("lasso", LassoCV(cv=5, random_state=0))
    ])
    pipe.fit(X, y)
    coef = pipe.named_steps["lasso"].coef_
    importance = pd.DataFrame({"feature": X.columns, "importance": np.abs(coef)}).sort_values("importance", ascending=False)
    return importance

# ========= Ensemble =========
def rmse(a, b):
    a = np.asarray(a); b = np.asarray(b)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() == 0:
        return np.inf
    return float(np.sqrt(np.mean((a[m]-b[m])**2)))

def ensemble_predictions(close: pd.Series,
                         prophet_fc: pd.DataFrame,
                         kalman_hist: pd.Series,
                         arima_hist: pd.Series,
                         lasso_hist: pd.Series,
                         kalman_fc: pd.DataFrame,
                         arima_fc: pd.DataFrame,
                         lasso_fc: pd.DataFrame,
                         backtest_window: int = 60) -> pd.DataFrame:

    close = close.astype(float)

    # 历史窗口
    max_window = max(10, min(backtest_window, int(len(close) * 0.5)))
    tail_idx = close.tail(max_window).index

    # 各模型历史一步预测价格
    prop_hist = prophet_fc.set_index("ds").reindex(close.index)["yhat"].tail(max_window)
    kal_hist = kalman_hist.reindex(close.index).tail(max_window)
    ari_hist = arima_hist.reindex(close.index).tail(max_window)
    las_hist = lasso_hist.reindex(close.index).tail(max_window)

    # RMSE
    e_prop = rmse(close.reindex(tail_idx).values, prop_hist.reindex(tail_idx).values)
    e_kal  = rmse(close.reindex(tail_idx).values, kal_hist.reindex(tail_idx).values)
    e_ari  = rmse(close.reindex(tail_idx).values, ari_hist.reindex(tail_idx).values)
    e_las  = rmse(close.reindex(tail_idx).values, las_hist.reindex(tail_idx).values)

    errs = np.array([e_prop, e_kal, e_ari, e_las], dtype=float)
    errs = np.where(np.isfinite(errs) & (errs > 0), errs, np.nan)
    inv = 1.0 / errs
    inv[np.isnan(inv)] = 0.0
    weights = (inv / inv.sum()) if inv.sum() > 0 else np.array([0.4, 0.3, 0.2, 0.1])

    # 未来合成
    dfs = [prophet_fc, kalman_fc, arima_fc, lasso_fc]
    H = min(len(d) for d in dfs) if dfs else 0
    horizon_dates = pd.bdate_range(close.index.max() + timedelta(days=1), periods=H)

    cols = ["yhat","yhat_lower","yhat_upper"]
    mats = []
    for d in dfs:
        dd = d.set_index("ds").reindex(horizon_dates).interpolate().ffill().bfill()
        mats.append(dd[cols].values)
    mats = np.stack(mats, axis=0)
    w = weights.reshape(-1,1,1)
    ens = (mats * w).sum(axis=0)

    out = pd.DataFrame(ens, columns=cols)
    out.insert(0, "ds", horizon_dates)
    out["rmse_prophet"] = errs[0]
    out["rmse_kalman"]  = errs[1]
    out["rmse_arima"]   = errs[2]
    out["rmse_lasso"]   = errs[3]
    out["weight_prophet"] = weights[0]
    out["weight_kalman"]  = weights[1]
    out["weight_arima"]   = weights[2]
    out["weight_lasso"]   = weights[3]
    out["backtest_window"] = max_window
    return out

# ========= Main =========
all_features, all_prophet, all_kalman_trend, all_ensemble, all_importances = [], [], [], [], []

macro_df = build_macro_frame(START_DATE)
sector_df = get_sector_etf_data(SECTOR_ETFS, START_DATE, END_DATE)

for tk in tqdm(TICKERS, desc="Processing"):
    try:
        feat = build_stock_features(tk, START_DATE, END_DATE)
        merged = feat.join(macro_df, how="left").join(sector_df, how="left")
        merged["EconCycle"] = classify_economic_cycle(merged)

        # 填充
        num_cols = merged.select_dtypes(include=[np.number]).columns
        merged[num_cols] = merged[num_cols].ffill().bfill()
        cat_cols = merged.select_dtypes(include=["category","object"]).columns
        merged[cat_cols] = merged[cat_cols].ffill().bfill()

        all_features.append(merged.assign(Ticker=tk))

        close = merged["Close"].astype(float)

        # 各模型
        fc_prophet = prophet_forecast(close, horizon_days=180)
        fc_prophet["Ticker"] = tk
        all_prophet.append(fc_prophet)

        kal_trend, kal_hist, fc_kalman = kalman_trend_and_forecast(close, horizon_days=180)
        kal_trend = kal_trend.to_frame(name="KalmanTrend")
        kal_trend["Ticker"] = tk
        all_kalman_trend.append(kal_trend.reset_index().rename(columns={"index":"Date"}))

        fc_arima, arima_hist = arima_returns_forecast(close, horizon_days=180)
        fc_lasso, lasso_hist = lasso_factor_forecast_rolling(merged, horizon_days=180)

        # 集成
        window = min(60, max(10, len(close)//2))
        fc_ens = ensemble_predictions(close,
                                      prophet_fc=fc_prophet,
                                      kalman_hist=kal_hist,
                                      arima_hist=arima_hist,
                                      lasso_hist=lasso_hist,
                                      kalman_fc=fc_kalman,
                                      arima_fc=fc_arima,
                                      lasso_fc=fc_lasso,
                                      backtest_window=window)
        fc_ens["Ticker"] = tk
        all_ensemble.append(fc_ens)

        # 特征重要性
        try:
            imp = compute_feature_importance(merged)
            imp["Ticker"] = tk
            all_importances.append(imp)
        except Exception as ee:
            print(f"[WARN] {tk} Lasso 重要性跳过：{ee}")

    except Exception as e:
        print(f"[ERROR] {tk} 失败: {e}")

print("Debug:", len(all_features), len(all_prophet), len(all_kalman_trend), len(all_ensemble), len(all_importances))

# ========= Save (CI-friendly: no git push here) =========
if all_features:
    pd.concat(all_features).reset_index().rename(columns={"index":"Date"}).to_csv("features.csv", index=False)
if all_prophet:
    pd.concat(all_prophet).to_csv("predictions_prophet.csv", index=False)
if all_kalman_trend:
    pd.concat(all_kalman_trend).to_csv("trend_kalman.csv", index=False)
if all_ensemble:
    pd.concat(all_ensemble).to_csv("predictions_ensemble.csv", index=False)
if all_importances:
    pd.concat(all_importances).to_csv("feature_importance.csv", index=False)

print("\nCSV files saved locally. Handle commit/push in GitHub Actions (recommended).")
print("\n=== 完成（Leakage修复 + 一步预测回测 + 交易日频率 + CI友好）===\n")
