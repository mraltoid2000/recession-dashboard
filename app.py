import os
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
st.write("App started successfully")
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
}

DEFAULT_WEIGHTS = {
    "10Y - 3M Yield Spread": 25,
    "Unemployment Rate": 20,
    "Initial Jobless Claims": 20,
    "Consumer Sentiment": 15,
    "Housing Permits": 10,
    "Industrial Production": 10,
}

START_DATE = "1988-01-01"


def require_api_key():
    if not FRED_API_KEY:
        st.write("DEBUG: API KEY IS MISSING")
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

