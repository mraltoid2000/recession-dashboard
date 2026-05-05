import os
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from fredapi import Fred

st.set_page_config(page_title="Recession Risk Dashboard", layout="wide")

FRED_API_KEY = os.getenv("FRED_API_KEY") or st.secrets.get("FRED_API_KEY", None)

SERIES = {
    "Recession Flag": "USREC",
    "10Y - 3M Yield Spread": "T10Y3M",
    "Unemployment Rate": "UNRATE",
    "Initial Jobless Claims": "ICSA",
    "Consumer Sentiment": "UMCSENT",
    "Housing Permits": "PERMIT",
    "Industrial Production": "INDPRO",
    "Gas Prices": "GASREGW",
    "Personal Saving Rate": "PSAVERT",
    "Consumer Credit": "TOTALSL",
}

DEFAULT_WEIGHTS = {
    "10Y - 3M Yield Spread": 20,
    "Unemployment Rate": 18,
    "Initial Jobless Claims": 18,
    "Consumer Sentiment": 12,
    "Housing Permits": 8,
    "Industrial Production": 8,
    "Gas Prices": 6,
    "Personal Saving Rate": 5,
    "Consumer Credit": 5,
}
START_DATE = "1988-01-01"


def require_api_key():
    if not FRED_API_KEY:
        st.error("Missing FRED_API_KEY. Add it to Streamlit secrets or your local environment.")
        st.stop()


@st.cache_data(ttl=24 * 60 * 60, show_spinner="Fetching latest FRED data...")
def load_fred_data(api_key: str) -> pd.DataFrame:
    fred = Fred(api_key=api_key)
    frames = []
    for label, series_id in SERIES.items():
        s = fred.get_series(series_id, observation_start=START_DATE)
        frames.append(s.rename(label))
    df = pd.concat(frames, axis=1).sort_index()
    df.index = pd.to_datetime(df.index)
    return df


def monthly_resample(df: pd.DataFrame) -> pd.DataFrame:
    monthly = pd.DataFrame(index=pd.date_range(df.index.min(), df.index.max(), freq="MS"))
    for col in df.columns:
        if col == "Recession Flag":
            monthly[col] = df[col].resample("MS").max()
        else:
            monthly[col] = df[col].resample("MS").mean()
    monthly = monthly.ffill()
    return monthly


def z_score(series: pd.Series, invert: bool = False, window: int = 120) -> pd.Series:
    rolling_mean = series.rolling(window, min_periods=36).mean()
    rolling_std = series.rolling(window, min_periods=36).std()
    z = (series - rolling_mean) / rolling_std
    if invert:
        z = -z
    return z.clip(-3, 3)


def logistic_score(z: pd.Series) -> pd.Series:
    return 100 / (1 + np.exp(-z))


def build_indicator_scores(df: pd.DataFrame) -> pd.DataFrame:
    scores = pd.DataFrame(index=df.index)

    # Lower / more negative yield spread is recessionary, so invert.
    scores["10Y - 3M Yield Spread"] = logistic_score(z_score(df["10Y - 3M Yield Spread"], invert=True))

    # Higher unemployment is recessionary.
    scores["Unemployment Rate"] = logistic_score(z_score(df["Unemployment Rate"], invert=False))

    # Higher jobless claims are recessionary.
    claims_monthly = df["Initial Jobless Claims"]
    scores["Initial Jobless Claims"] = logistic_score(z_score(claims_monthly, invert=False))

    # Lower sentiment is recessionary, so invert.
    scores["Consumer Sentiment"] = logistic_score(z_score(df["Consumer Sentiment"], invert=True))

    # Lower permits are recessionary, so invert.
    scores["Housing Permits"] = logistic_score(z_score(df["Housing Permits"], invert=True))

    # Lower industrial production is recessionary, so invert.
    scores["Industrial Production"] = logistic_score(z_score(df["Industrial Production"], invert=True))

    # Higher gas prices are recessionary because they pressure consumers.
    scores["Gas Prices"] = logistic_score(z_score(df["Gas Prices"], invert=False))

    # Lower saving rate is recessionary, so invert.
    scores["Personal Saving Rate"] = logistic_score(z_score(df["Personal Saving Rate"], invert=True))

    # Higher consumer credit can indicate debt stress, so treat higher values as higher risk.
    scores["Consumer Credit"] = logistic_score(z_score(df["Consumer Credit"], invert=False))
    
    return scores.dropna(how="all")


def weighted_metric(scores: pd.DataFrame, weights: dict) -> pd.Series:
    weight_series = pd.Series(weights, dtype=float)
    if weight_series.sum() == 0:
        weight_series[:] = 1
    normalized_weights = weight_series / weight_series.sum()
    return scores[normalized_weights.index].mul(normalized_weights, axis=1).sum(axis=1)


def plot_metric(metric: pd.Series, recession: pd.Series):
    fig = go.Figure()

    rec = recession.reindex(metric.index).fillna(0)
    in_rec = False
    start = None
    for idx, val in rec.items():
        if val == 1 and not in_rec:
            start = idx
            in_rec = True
        if val == 0 and in_rec:
            fig.add_vrect(x0=start, x1=idx, fillcolor="gray", opacity=0.2, line_width=0)
            in_rec = False
    if in_rec:
        fig.add_vrect(x0=start, x1=metric.index[-1], fillcolor="gray", opacity=0.2, line_width=0)

    fig.add_trace(go.Scatter(x=metric.index, y=metric, name="Composite Risk Metric", line=dict(width=3)))
    fig.add_hline(y=70, line_dash="dash", annotation_text="High-risk threshold")
    fig.add_hline(y=50, line_dash="dot", annotation_text="Neutral")
    fig.update_layout(height=520, yaxis_title="Risk score, 0–100", xaxis_title="Date")
    return fig


def plot_indicators(scores: pd.DataFrame):
    fig = go.Figure()
    for col in scores.columns:
        fig.add_trace(go.Scatter(x=scores.index, y=scores[col], name=col))
    fig.update_layout(height=520, yaxis_title="Indicator risk score, 0–100", xaxis_title="Date")
    return fig


def latest_table(raw: pd.DataFrame, scores: pd.DataFrame, weights: dict) -> pd.DataFrame:
    rows = []
    for name in scores.columns:
        latest_score = scores[name].dropna().iloc[-1]
        latest_raw = raw[name].dropna().iloc[-1]
        latest_date = raw[name].dropna().index[-1].date()
        rows.append({
            "Indicator": name,
            "Latest raw value": round(float(latest_raw), 3),
            "Latest date": latest_date,
            "Risk score": round(float(latest_score), 1),
            "Weight": weights[name],
        })
    return pd.DataFrame(rows).sort_values("Risk score", ascending=False)


require_api_key()

st.title("Recession Risk Dashboard")
st.caption("Composite metric built from historical FRED indicators. Shaded regions are NBER recession periods from FRED's USREC series.")

raw_daily = load_fred_data(FRED_API_KEY)
raw = monthly_resample(raw_daily)
scores = build_indicator_scores(raw)

with st.sidebar:
    st.header("Indicator weights")
    st.write("Adjust how much each indicator contributes to the composite metric.")
    weights = {}
    for indicator, default in DEFAULT_WEIGHTS.items():
        weights[indicator] = st.slider(indicator, 0, 100, default, 1)

    st.divider()
    start_year = st.slider("Chart start year", 1988, date.today().year, 1995)

metric = weighted_metric(scores, weights)
metric = metric[metric.index.year >= start_year]
scores_display = scores[scores.index.year >= start_year]
recession_display = raw["Recession Flag"][raw.index.year >= start_year]

latest_metric = metric.dropna().iloc[-1]
latest_date = metric.dropna().index[-1].date()

col1, col2, col3 = st.columns(3)
col1.metric("Current composite risk", f"{latest_metric:.1f} / 100")
col2.metric("Latest metric date", str(latest_date))
col3.metric("Active recession flag", "Yes" if raw["Recession Flag"].dropna().iloc[-1] == 1 else "No")

st.plotly_chart(plot_metric(metric, recession_display), use_container_width=True)

st.subheader("Indicator risk scores")
st.plotly_chart(plot_indicators(scores_display), use_container_width=True)

st.subheader("Latest indicator readings")
st.dataframe(latest_table(raw, scores, weights), use_container_width=True, hide_index=True)

st.info(
    "This is a signal dashboard, not a forecast. The initial scoring model uses rolling z-scores and a logistic transform. "
    "The next improvement is to calibrate weights against the last three recessions using lead-time and false-positive analysis."
)
