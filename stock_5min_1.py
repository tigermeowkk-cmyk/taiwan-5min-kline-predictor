import os
import time
import datetime
import requests
import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
from scipy.signal import find_peaks

st.set_page_config(page_title="股票5分盤AI預測工具", layout="wide")

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
FUGLE_URL = "https://api.fugle.tw/marketdata/v1.0/stock"
API_CALL_DELAY_SECONDS = 0.3

# ==========================================
# 底層：FinMind 呼叫
# 這支程式直接打 REST API，不透過 FinMind 套件（不用另外 pip install FinMind）。
#
# 重要：TaiwanStockKBar／TaiwanFuturesTick 這兩個高頻資料集，FinMind 只要一次帶 end_date
# 就會回 400（"size too large...end_date parameter need be none"），要抓多天只能一天一天
# 分開打。這點在原本的版本裡沒被發現，之前只抓 1~5 天大部分時候剛好在單日限制內才沒爆出來，
# 一旦拉長回溯天數其實整段都會抓空。這裡改成明確逐日迴圈抓取。
# ==========================================

def _finmind_get_day(dataset, data_id, day, token):
    """單日查詢，用於 TaiwanStockKBar / TaiwanFuturesTick 這類分K/逐筆資料集。"""
    params = {"dataset": dataset, "data_id": data_id, "start_date": day.isoformat(), "token": token}
    try:
        r = requests.get(FINMIND_URL, params=params, timeout=25)
        j = r.json()
        return j.get("data") or []
    except Exception:
        return []


def _finmind_get_range(dataset, data_id, start, end, token):
    """範圍查詢，用於 TaiwanFuturesDaily／USStockPrice／TaiwanExchangeRate 這類日頻資料集。"""
    params = {
        "dataset": dataset, "data_id": data_id,
        "start_date": start.isoformat(), "end_date": end.isoformat(), "token": token,
    }
    try:
        r = requests.get(FINMIND_URL, params=params, timeout=25)
        j = r.json()
        return j.get("data") or []
    except Exception:
        return []


def _trading_days(start, end):
    days, d = [], start
    while d <= end:
        if d.weekday() < 5:  # 週末必然沒交易；國定假日抓回來會是空資料，不影響結果
            days.append(d)
        d += datetime.timedelta(days=1)
    return days


def get_front_month_contract(prefix, as_of=None):
    """近月合約代碼（大台/小台通用，結算日邏輯相同，差別只在代碼前綴）。"""
    as_of = as_of or datetime.date.today()
    year, month = as_of.year, as_of.month
    first_day = datetime.date(year, month, 1)
    first_wed = first_day + datetime.timedelta(days=(2 - first_day.weekday() + 7) % 7)
    settlement_day = first_wed + datetime.timedelta(days=14)
    if as_of > settlement_day:
        month += 1
        if month > 12:
            month = 1
            year += 1
    return f"{prefix}{year}{month:02d}"


# ==========================================
# 資料抓取：個股 / 期貨 5 分K
# ==========================================

@st.cache_data(ttl=60, show_spinner=False)
def load_stock_5min_fugle(stock_id, api_key):
    """
    個股改用 Fugle 即時行情（免費方案就有）：一次呼叫直接拿近 30 天原生 5 分K，
    資料新鮮度比 FinMind（歷史/延遲資料）好很多，符合當沖需要的即時性。
    Fugle 這個分K端點不能指定日期範圍，固定回傳近 30 天，不需要像 FinMind 那樣逐日迴圈抓。
    """
    try:
        r = requests.get(
            f"{FUGLE_URL}/historical/candles/{stock_id}",
            params={"timeframe": "5"},
            headers={"X-API-KEY": api_key},
            timeout=20,
        )
        rows = r.json().get("data") or []
    except Exception:
        rows = []

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values("date").reset_index(drop=True)
    df["is_night_session"] = 0
    return df[["date", "open", "high", "low", "close", "volume", "is_night_session"]]


@st.cache_data(ttl=300, show_spinner=False)
def load_futures_5min(prefix, start, end, token):
    """prefix: 'TX'（大台）或 'MTX'（小台）。日盤＋夜盤都保留、串成連續序列。"""
    frames = []
    for day in _trading_days(start, end):
        rows = _finmind_get_day("TaiwanFuturesTick", prefix, day, token)
        if not rows:
            time.sleep(API_CALL_DELAY_SECONDS)
            continue

        df_day = pd.DataFrame(rows)
        # 只留純月合約 (YYYYMM，6碼數字)，排除週合約 (202607W4) 與價差合約 (202608/202609)
        df_day = df_day[df_day["contract_date"].str.fullmatch(r"\d{6}")]
        if df_day.empty:
            time.sleep(API_CALL_DELAY_SECONDS)
            continue

        near_month = df_day["contract_date"].min()
        df_day = df_day[df_day["contract_date"] == near_month]
        frames.append(df_day)
        time.sleep(API_CALL_DELAY_SECONDS)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    df_5m = df["price"].resample("5min").ohlc()
    df_5m["volume"] = df["volume"].resample("5min").sum()
    df_5m = df_5m.dropna().reset_index()

    hour = df_5m["date"].dt.hour
    df_5m["is_night_session"] = ((hour >= 15) | (hour < 5)).astype(int)
    return df_5m


@st.cache_data(ttl=3600, show_spinner=False)
def load_daily_macro(start, end, token, include_tx_futures):
    """
    給模型當「慢變數」的大盤前導特徵：美股(TSM ADR 代理)、美元兌台幣，都是日頻、當天用前一筆值鋪滿。
    台指期走勢只有在預測「個股」時才加（預測期貨本身時，期貨自己的走勢就是預測目標，不該把自己當特徵）。
    任何一項抓不到就跳過那一項，不讓單一資料源失敗拖垮其他特徵。
    """
    frames = []

    try:
        rows = _finmind_get_range("USStockPrice", "TSM", start, end, token)
        if rows:
            us = pd.DataFrame(rows).sort_values("date")
            price_col = "Adj_Close" if "Adj_Close" in us.columns else "Close" if "Close" in us.columns else "close"
            us["us_close_pct"] = (us[price_col] - us["open" if "open" in us.columns else "Open"]) / us["open" if "open" in us.columns else "Open"]
            us["pure_date"] = (pd.to_datetime(us["date"]) + pd.Timedelta(days=1)).dt.date
            frames.append(us[["pure_date", "us_close_pct"]])
    except Exception:
        pass

    try:
        rows = _finmind_get_range("TaiwanExchangeRate", "USD", start, end, token)
        if rows:
            fx = pd.DataFrame(rows).sort_values("date")
            fx["_mid"] = (fx["spot_buy"] + fx["spot_sell"]) / 2
            fx["usdtwd_chg"] = fx["_mid"].pct_change()
            fx["pure_date"] = pd.to_datetime(fx["date"]).dt.date
            frames.append(fx[["pure_date", "usdtwd_chg"]].dropna())
    except Exception:
        pass

    if include_tx_futures:
        try:
            rows = _finmind_get_range("TaiwanFuturesDaily", "TX", start, end, token)
            if rows:
                tx = pd.DataFrame(rows)
                tx = tx[tx["contract_date"].str.fullmatch(r"\d{6}")]
                near = tx.groupby("date")["contract_date"].min().rename("_near").reset_index()
                tx = tx.merge(near, on="date")
                tx = tx[tx["contract_date"] == tx["_near"]]

                day_s = tx[tx["trading_session"] == "position"][["date", "close"]].rename(columns={"close": "_day"})
                night_s = tx[tx["trading_session"] == "after_market"][["date", "close"]].rename(columns={"close": "_night"})
                merged = night_s.merge(day_s, on="date", how="inner").sort_values("date")
                merged["_prev_day"] = merged["_day"].shift(1)
                merged["tx_night_chg"] = merged["_night"] / merged["_prev_day"] - 1
                merged["pure_date"] = pd.to_datetime(merged["date"]).dt.date
                frames.append(merged[["pure_date", "tx_night_chg"]].dropna())
        except Exception:
            pass

    if not frames:
        return pd.DataFrame()

    macro = frames[0]
    for f in frames[1:]:
        macro = macro.merge(f, on="pure_date", how="outer")
    return macro.sort_values("pure_date").reset_index(drop=True)


# ==========================================
# 前台
# ==========================================

st.markdown("""
<style>
.mobile-hint { display: none; }
@media (max-width: 768px) {
    .mobile-hint {
        display: block;
        background-color: #fffbeb;
        color: #d97706;
        padding: 10px 14px;
        border-radius: 8px;
        margin-bottom: 12px;
        font-weight: 600;
    }
}
</style>
""", unsafe_allow_html=True)

st.title("📈 股票5分盤AI預測工具")
st.markdown("<div class='mobile-hint'>📱 手機版用戶：請點擊左上角 <strong>「&gt;」</strong> 符號展開側邊欄，開始設定預測參數！</div>", unsafe_allow_html=True)

# ⚖️ 使用前先跳出投資免責聲明，沒按同意就不能操作
@st.dialog("⚖️ 投資免責聲明", width="large")
def show_disclaimer_modal():
    st.warning("⚠️ **在使用本工具之前，請務必閱讀以下投資免責聲明：**")
    st.markdown("""
    * 本工具所有預測結果，皆為程式演算法根據歷史資料自動運算之產出。
    * ❌ **不構成**任何買賣與投資操作之要約、承諾或建議。
    * ❌ **不保證**未來獲利或預測之準確性，過去績效不代表未來表現。

    投資人應獨立判斷、審慎評估，並自負最終投資盈虧風險。
    """)
    st.write("")
    if st.button("✅ 我已閱讀並完全同意上述聲明，進入系統", type="primary", use_container_width=True):
        st.session_state.disclaimer_accepted = True
        st.rerun()

if "disclaimer_accepted" not in st.session_state:
    st.session_state.disclaimer_accepted = False
if not st.session_state.disclaimer_accepted:
    show_disclaimer_modal()

st.sidebar.header("參數設定")
product_type = st.sidebar.selectbox("商品類型", ["個股", "大台指期 (TX)", "小台指期 (MTX)"], disabled=not st.session_state.disclaimer_accepted)
stock_code = st.sidebar.text_input("股票代碼", value="2330", disabled=not st.session_state.disclaimer_accepted) if product_type == "個股" else None
lookback_days = st.sidebar.slider("回溯天數（訓練資料量）", min_value=5, max_value=20, value=10, disabled=not st.session_state.disclaimer_accepted)
if product_type == "個股":
    st.sidebar.caption("💡 個股走 Fugle 即時行情，資料新鮮度接近即時；固定抓近 30 天，這裡的天數只是拿來裁切訓練窗口。")
else:
    st.sidebar.caption("💡 期貨目前仍用 FinMind 歷史逐筆資料（非即時），天數拉長會明顯變慢，建議先從 5~10 天開始測試。")

if st.sidebar.button("開始執行預測", disabled=not st.session_state.disclaimer_accepted):
    # 按下去之後自動收合側邊欄，讓手機版直接看到預測結果，不用使用者自己再手動收起來
    st.components.v1.html("""
    <script>
        window.parent.document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', keyCode: 27}));
        var closeBtns = window.parent.document.querySelectorAll('button[kind="headerNoPadding"]');
        if (closeBtns.length > 0) {
            closeBtns.forEach(btn => btn.click());
        } else {
            var altBtn = window.parent.document.querySelector('[data-testid="stSidebarCollapseButton"]');
            if (altBtn) { altBtn.click(); }
        }
    </script>
    """, height=0, width=0)

    finmind_token = os.environ.get("FINMIND_TOKEN")
    fugle_key = os.environ.get("FUGLE_API_KEY")

    if product_type == "個股" and not fugle_key:
        st.error("🔑 找不到 Fugle API Key！請在 Settings > Secrets 設定 FUGLE_API_KEY")
        st.stop()
    if not finmind_token:
        st.error("🔑 找不到 FinMind Token！請在 Settings > Secrets 設定 FINMIND_TOKEN")
        st.stop()

    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=lookback_days)

    with st.spinner("🤖 正在抓取資料並訓練模型..."):
        if product_type == "個股":
            df = load_stock_5min_fugle(stock_code, fugle_key)
            if not df.empty:
                bars_per_day_estimate = 54  # 09:00~13:30 一天約 54 根 5 分K
                df = df.tail(lookback_days * bars_per_day_estimate).reset_index(drop=True)
            label = stock_code
        elif product_type.startswith("大台"):
            df = load_futures_5min("TX", start_date, end_date, finmind_token)
            label = "大台指期 (TX)"
        else:
            df = load_futures_5min("MTX", start_date, end_date, finmind_token)
            label = "小台指期 (MTX)"

        if df.empty:
            st.warning("⚠️ 查無資料，請確認代碼是否正確，或拉長回溯天數再試一次。")
            st.stop()

        df = df.sort_values("date").reset_index(drop=True)

        # ---- 特徵工程 ----
        df["open_close_pct"] = (df["close"] - df["open"]) / df["open"]
        df["high_low_pct"] = (df["high"] - df["low"]) / df["low"]

        df["ma_5"] = df["close"].rolling(5).mean()
        df["ma_20"] = df["close"].rolling(20).mean()
        df["ma_50"] = df["close"].rolling(50).mean()
        df["ma_80"] = df["close"].rolling(80).mean()
        df["v_ma_5"] = df["volume"].rolling(5).mean()

        df["bias_5"] = (df["close"] - df["ma_5"]) / df["ma_5"]
        df["bias_20"] = (df["close"] - df["ma_20"]) / df["ma_20"]

        delta = df["close"].diff()
        up, down = delta.clip(lower=0), -1 * delta.clip(upper=0)
        ema_up = up.ewm(com=13, adjust=False).mean()
        ema_down = down.ewm(com=13, adjust=False).mean()
        df["rsi_14"] = 100 - (100 / (1 + (ema_up / ema_down)))

        ema_12 = df["close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        df["minutes_since_open"] = ((df["date"].dt.hour - 9) * 60 + df["date"].dt.minute).clip(lower=0)

        # ---- 大盤前導特徵（慢變數，日頻 ffill 鋪到當天每根5分K）----
        macro = load_daily_macro(start_date, end_date, finmind_token, include_tx_futures=(product_type == "個股"))
        df["pure_date"] = df["date"].dt.date
        macro_cols = []
        if not macro.empty:
            df = df.merge(macro, on="pure_date", how="left")
            macro_cols = [c for c in macro.columns if c != "pure_date"]
            for col in macro_cols:
                df[col] = df[col].ffill().fillna(0)

        feature_cols = [
            "open_close_pct", "high_low_pct", "bias_5", "bias_20",
            "volume", "v_ma_5", "rsi_14", "macd", "macd_hist", "minutes_since_open",
        ] + macro_cols
        if product_type != "個股":
            feature_cols.append("is_night_session")

        # ---- 建立標籤、分離最新一筆待預測資料 ----
        df["next_close"] = df["close"].shift(-1)
        latest_data = df.iloc[[-1]].copy()
        df_train_set = df.dropna(subset=feature_cols + ["next_close"]).reset_index(drop=True)

        if len(df_train_set) < 60:
            st.warning("⚠️ 訓練資料量不足（可能回溯天數太短或該商品當時交易清淡），請拉長回溯天數再試一次。")
            st.stop()

        df_train_set["target"] = (df_train_set["next_close"] > df_train_set["close"]).astype(int)

        X = df_train_set[feature_cols]
        y = df_train_set["target"]

        # ---- 模型訓練：時間序列切分，不打亂順序 ----
        split_idx = int(len(df_train_set) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, eval_metric="logloss")
        model.fit(X_train, y_train)

        acc = accuracy_score(y_test, model.predict(X_test)) if len(X_test) >= 5 else None

        latest_features = latest_data[feature_cols].fillna(0)
        next_pred = model.predict(latest_features)[0]
        next_pred_proba = model.predict_proba(latest_features)[0]
        ai_confidence = next_pred_proba[1] if next_pred == 1 else next_pred_proba[0]

        # ----------------- 🎯 型態偵測 ＋ AI 雙重認證 -----------------
        lookback = 40
        recent_data = df_train_set.tail(lookback).reset_index(drop=True)
        closes = recent_data["close"].values

        valleys, _ = find_peaks(-closes, distance=3)
        peaks, _ = find_peaks(closes, distance=3)

        alert_status = "info"
        alert_msg = "ℹ️ **當前市場監控**：暫未觸發明確的 W底黃金進場 或 型態破敗退場 機制。請維持紀律操作。"

        if len(valleys) >= 2:
            l_foot, r_foot = valleys[-2], valleys[-1]
            l_foot_p, r_foot_p = closes[l_foot], closes[r_foot]
            ma20_r = recent_data["ma_20"].iloc[r_foot]
            ma50_r = recent_data["ma_50"].iloc[r_foot]
            lower_bound, upper_bound = min(ma20_r, ma50_r), max(ma20_r, ma50_r)

            if (r_foot_p > l_foot_p) and (lower_bound <= r_foot_p <= upper_bound):
                if next_pred == 1:
                    alert_status = "success"
                    alert_msg = f"🟢 **【安全進場訊號】** 5分K偵測到標準 W 底（右腳高於左腳），且回測落於 MA20 ~ MA50 的關鍵防守區！**此時後台 AI 模型同步強烈看漲（信心度: {ai_confidence*100:.1f}%）**，技術面與數據預測達成高度共識，**此處可安全進場**。"
                else:
                    alert_status = "warning"
                    alert_msg = f"⚠️ **【進場風險提示】** 技術面雖然出現 W 底且落於 MA20 ~ MA50 支撐區，**但 AI 後台模型偵測到潛在轉弱風險（下跌/盤整機率: {ai_confidence*100:.1f}%）**。此型態極有可能為誘多的「假突破」，**不建議此處冒險進場**。"

        if len(valleys) >= 2 and alert_status == "info":
            l_foot, r_foot = valleys[-2], valleys[-1]
            l_foot_p, r_foot_p = closes[l_foot], closes[r_foot]
            ma50_r = recent_data["ma_50"].iloc[r_foot]
            ma80_r = recent_data["ma_80"].iloc[r_foot]
            lower_bound, upper_bound = min(ma50_r, ma80_r), max(ma50_r, ma80_r)

            if (r_foot_p < l_foot_p) and (lower_bound <= r_foot_p <= upper_bound):
                if next_pred == 0:
                    alert_status = "error"
                    alert_msg = f"🚨 **【安全退場訊號】** 5分K走出轉弱型態（右腳低於左腳），且價格已沉淪至 MA50 ~ MA80 的弱勢區間。**同時 AI 後台模型亦全面看空（看跌信心度: {ai_confidence*100:.1f}%）**，均線防守全面失守，**強烈建議現股多單退場或進行避險**。"
                else:
                    alert_status = "warning"
                    alert_msg = f"⚠️ **【退場風險提示】** 雖然價格落入 MA50 ~ MA80 且右腳偏低，但 **AI 後台模型預估此處即將迎來短線反彈（看漲信心度: {ai_confidence*100:.1f}%）**，目前可能屬於主力的洗盤破底翻，**建議在此處暫緩殺低，觀察下一根K線是否站穩**。"

        if alert_status == "success":
            st.success(alert_msg)
        elif alert_status == "error":
            st.error(alert_msg)
        elif alert_status == "warning":
            st.warning(alert_msg)
        else:
            st.info(alert_msg)

        status_text = "🔴 上漲" if next_pred == 1 else "🔵 下跌 / 盤整"
        st.subheader(f"🤖 AI 模型即時預測方向 (下一根5分K)：{status_text} (預測信心度: {ai_confidence*100:.1f}%)")

        # ---- 誠實揭露：樣本外準確率 ----
        if acc is not None:
            st.metric(
                "模型近期樣本外準確率 (Accuracy)", f"{acc*100:.2f}%",
                help=f"用時間序列切分的最後 {len(X_test)} 根K棒測試，樣本數偏小時這個數字本身會有明顯隨機波動，僅供參考，不代表未來績效。",
            )
            if len(X_test) < 30:
                st.caption(f"⚠️ 測試樣本只有 {len(X_test)} 根K棒，準確率數字的可信度有限，建議拉長回溯天數觀察是否穩定。")
        else:
            st.caption("⚠️ 測試集資料不足，這次無法計算樣本外準確率，結果請更謹慎看待。")

        st.markdown("---")

        st.subheader(f"{label} 近期 5 分鐘 K 線走勢")
        fig = go.Figure(data=[go.Candlestick(
            x=df_train_set["date"].iloc[split_idx:],
            open=df_train_set["open"].iloc[split_idx:],
            high=df_train_set["high"].iloc[split_idx:],
            low=df_train_set["low"].iloc[split_idx:],
            close=df_train_set["close"].iloc[split_idx:],
            name="K線",
        )])
        fig.update_layout(xaxis_rangeslider_visible=False, height=500)
        st.plotly_chart(fig, use_container_width=True)

        # 特徵重要性只留後台看（Render 的 log），不顯示在使用者畫面上
        importance_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": model.feature_importances_,
        }).sort_values("Importance", ascending=False)
        print(f"[{label}] 特徵重要性：\n{importance_df.to_string(index=False)}")
