import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import numpy as np
import streamlit as st
from mplfinance.original_flavor import candlestick_ohlc

# Parameters (defaults)
DEFAULT_TICKER = "AAPL"
DEFAULT_START = dt.date(2023, 1, 1)
DEFAULT_END = dt.date.today()
RSI_PERIOD = 14

def download_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"No data downloaded for {ticker}")
    if 'Adj Close' not in df.columns:
        df['Adj Close'] = df['Close']
    df.index = pd.to_datetime(df.index)
    df = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].dropna()
    df = df.astype(float)
    return df

def compute_indicators(df):
    out = pd.DataFrame(index=df.index)
    close = df['Adj Close']
    high = df['High']
    low = df['Low']
    vol = df['Volume'] if 'Volume' in df.columns else pd.Series(0, index=df.index)

    # RSI
    delta = close.diff(1)
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(window=RSI_PERIOD, min_periods=RSI_PERIOD).mean()
    avg_loss = loss.rolling(window=RSI_PERIOD, min_periods=RSI_PERIOD).mean()
    rs = avg_gain / avg_loss
    out['RSI'] = 100 - (100 / (1 + rs))

    # MACD (12,26) and signal 9
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    out['MACD'] = macd
    out['MACD_signal'] = signal
    out['MACD_hist'] = macd - signal

    # ROC and raw momentum (period 12)
    ROC_PERIOD = 12
    out['ROC'] = close.pct_change(ROC_PERIOD) * 100
    out['Momentum'] = close - close.shift(ROC_PERIOD)

    # Stochastic Oscillator (14,3)
    STO_PERIOD = 14
    K_smooth = 3
    lowest_low = low.rolling(window=STO_PERIOD).min()
    highest_high = high.rolling(window=STO_PERIOD).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    percent_k = 100 * (close - lowest_low) / denom
    if isinstance(percent_k, pd.DataFrame):
        percent_k = percent_k.iloc[:, 0]
    elif isinstance(percent_k, np.ndarray):
        percent_k = pd.Series(percent_k, index=out.index, name='%K')
    out['%K'] = percent_k.astype(float)
    out['%D'] = out['%K'].rolling(window=K_smooth).mean()

    # CCI (20)
    CCI_N = 20
    tp = (high + low + close) / 3.0
    sma_tp = tp.rolling(window=CCI_N).mean()
    mad = tp.rolling(window=CCI_N).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (tp - sma_tp) / (0.015 * mad)
    if isinstance(cci, pd.DataFrame):
        cci = cci.iloc[:, 0]
    out['CCI'] = cci.astype(float)

    # MFI (14) Money Flow Index (volume-aware)
    MFI_N = 14
    typical_price = tp
    money_flow = typical_price * vol
    tp_diff = typical_price.diff()
    pos_flow = money_flow.where(tp_diff > 0, 0.0)
    neg_flow = money_flow.where(tp_diff < 0, 0.0)
    pos_mf = pos_flow.rolling(window=MFI_N).sum()
    neg_mf = neg_flow.rolling(window=MFI_N).sum().abs()
    mfr = pos_mf / (neg_mf.replace(0, np.nan))
    mfi = 100 - (100 / (1 + mfr))
    if isinstance(mfi, pd.DataFrame):
        mfi = mfi.iloc[:, 0]
    out['MFI'] = mfi.astype(float)

    # OBV (On-Balance Volume) and a simple predictive slope (linear trend of OBV)
    obv = vol.copy()
    obv[:] = 0
    obv[close > close.shift(1)] = vol[close > close.shift(1)]
    obv[close < close.shift(1)] = -vol[close < close.shift(1)]
    out['OBV'] = obv.cumsum()
    out['OBV_sma20'] = out['OBV'].rolling(window=20).mean()
    out['OBV_slope'] = out['OBV'].diff(5)

    return out

# === Streamlit UI ===
st.set_page_config(page_title='RSI & Technical Indicators', layout='wide')

# Sticky/frozen dashboard title at the very top
st.markdown("""
    <style>
    .sticky-title {
        position: fixed;
        top: 1.5rem;
        left: 0;
        width: 100vw;
        z-index: 10000;
        background: #111;
        padding: 0.5rem 0 0.5rem 0;
        margin-bottom: 0;
        border-bottom: 2px solid #00ff00;
    }
    .stApp {
        padding-top: 5.5rem !important;
    }
    body, .stApp { background: #111 !important; color: #00ff00 !important; font-family: 'Fira Mono', monospace !important; }
    .stButton>button, .stDownloadButton>button {
        background: #222 !important;
        color: #00ff00 !important;
        border: 1.5px solid #00ff00 !important;
        font-weight: bold;
        letter-spacing: 1px;
        box-shadow: 0 0 8px #00ff0044;
        transition: background 0.2s;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        background: #00ff00 !important;
        color: #111 !important;
        border: 1.5px solid #00ff00 !important;
        box-shadow: 0 0 16px #00ff0099;
    }
    .stTextInput>div>div>input, .stDateInput>div>input {
        background: #222 !important;
        color: #00ff00 !important;
        border: 1.5px solid #00ff00 !important;
        font-family: 'Fira Mono', monospace !important;
        font-size: 0.75rem;
    }
    .stDataFrame, .stTable {
        background: #111 !important;
        color: #00ff00 !important;
        border: 1px solid #00ff00 !important;
    }
    </style>
    <div class="sticky-title">
        <h1 style="color:#00ff00; text-align:center; text-shadow:0 0 12px #00ff00; margin-bottom:0;">
            RSI & Technical Indicator Dashboard
        </h1>
    </div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([2,2,2])
with col1:
    ticker = st.text_input('Ticker', value=DEFAULT_TICKER).upper()
with col2:
    start = st.date_input('Start Date', value=DEFAULT_START)
with col3:
    end = st.date_input('End Date', value=DEFAULT_END)

run = st.button('Generate')

if run:
    try:
        with st.spinner('Fetching and computing...'):
            df = download_data(ticker, start, end + dt.timedelta(days=1))
            if df.empty:
                st.error('No data found for the given ticker/date range.')
            else:
                indicators = compute_indicators(df)
                st.markdown('---')
                st.markdown(f"### {ticker} Candlestick Chart")
                # ---- Candlestick Chart ----
                candle_df = df.copy().reset_index()
                date_col = 'index' if 'index' in candle_df.columns else 'Date'
                candle_df['DateNum'] = candle_df[date_col].map(mdates.date2num)
                ohlc = candle_df[['DateNum', 'Open', 'High', 'Low', 'Close']].values

                fig_candle, ax_candle = plt.subplots(figsize=(10, 3), dpi=2000)  # Increased DPI for sharpness
                candlestick_ohlc(ax_candle, ohlc, colorup='#00ff00', colordown='#ff1744', width=0.2)
                ax_candle.xaxis_date()
                ax_candle.set_title(f"{ticker} Candlestick Chart", color='white')
                ax_candle.set_facecolor("black")
                ax_candle.figure.set_facecolor("#121212")
                ax_candle.tick_params(axis='x', colors='white')
                ax_candle.tick_params(axis='y', colors='white')
                ax_candle.grid(True, color='#444444')
                fig_candle.autofmt_xdate()
                fig_candle.tight_layout()  # Add this for better spacing
                st.pyplot(fig_candle, clear_figure=True)

                st.markdown(f"### {ticker} Technical Indicator Charts")
                # ---- Technical Indicator Subplots ----
                plt.close('all')
                plt.style.use('dark_background')
                nrows = 4
                ncols = 2
                fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, figsize=(12, 8))  # Reduced size
                axes = axes.flatten()
                price_ax, rsi_ax, macd_ax, sto_ax, roc_ax, cci_ax, mfi_ax, obv_ax = axes

                price_ax.plot(df.index, df['Adj Close'], color='lightgray', label='Adj Close')
                price_ax.set_title(f"{ticker} Price", color='white')
                price_ax.grid(True, color='#444444')

                rsi_ax.plot(indicators.index, indicators['RSI'], color='lightgray')
                rsi_ax.axhline(30, color='#00ff00', linestyle='--', alpha=0.6)
                rsi_ax.axhline(70, color='#ff1744', linestyle='--', alpha=0.6)
                rsi_ax.set_ylabel('RSI')

                macd_ax.plot(indicators.index, indicators['MACD'], color='#00ff00', label='MACD')
                macd_ax.plot(indicators.index, indicators['MACD_signal'], color='#ff1744', label='Signal')
                hist = indicators['MACD_hist'].fillna(0)
                macd_ax.bar(hist.index, hist.values, color=(hist > 0).map({True: '#00ff00', False: '#ff1744'}), alpha=0.6)
                macd_ax.legend(loc='upper left')
                macd_ax.set_ylabel('MACD')

                sto_ax.plot(indicators.index, indicators['%K'], color='lightgray', label='%K')
                sto_ax.plot(indicators.index, indicators['%D'], color='#00ff00', label='%D')
                sto_ax.axhline(20, color='#00ff00', linestyle='--', alpha=0.5)
                sto_ax.axhline(80, color='#ff1744', linestyle='--', alpha=0.5)
                sto_ax.set_ylabel('Stoch')
                sto_ax.legend(loc='upper left')

                roc_ax.plot(indicators.index, indicators['ROC'], color='lightgray', label='ROC (%)')
                roc_ax.plot(indicators.index, indicators['Momentum'], color='#00ff00', label='Momentum')
                roc_ax.axhline(0, color='#666666', linestyle='--')
                roc_ax.set_ylabel('ROC / Mom')
                roc_ax.legend(loc='upper left')

                cci_ax.plot(indicators.index, indicators['CCI'], color='lightgray')
                cci_ax.axhline(100, color='#ff1744', linestyle='--')
                cci_ax.axhline(-100, color='#00ff00', linestyle='--')
                cci_ax.set_ylabel('CCI')

                mfi_ax.plot(indicators.index, indicators['MFI'], color='lightgray')
                mfi_ax.axhline(20, color='#00ff00', linestyle='--')
                mfi_ax.axhline(80, color='#ff1744', linestyle='--')
                mfi_ax.set_ylabel('MFI')

                obv_ax.plot(indicators.index, indicators['OBV'], color='lightgray', label='OBV')
                obv_ax.plot(indicators.index, indicators['OBV_sma20'], color='#00ff00', label='OBV_sma20')
                obv_ax.bar(indicators.index, indicators['OBV_slope'].fillna(0), color=(indicators['OBV_slope'] > 0).map({True: '#00ff00', False: '#ff1744'}), alpha=0.6)
                obv_ax.set_ylabel('OBV')
                obv_ax.legend(loc='upper left')

                for ax in axes:
                    ax.set_facecolor('black')
                    ax.tick_params(axis='x', colors='white')
                    ax.tick_params(axis='y', colors='white')

                plt.tight_layout()
                st.pyplot(fig, clear_figure=True)
    except Exception as e:
        st.error(f"Error: {e}")
