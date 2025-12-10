import streamlit as st
import datetime as dt
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_finance import candlestick_ohlc

st.set_page_config(
    page_title="Terminal Chart Suite",
    page_icon="novus_logo.png",
    layout="wide"
)

st.markdown("""
    <style>
    body, .stApp {
        background: #111 !important;
        color: #00ff00 !important;
        font-family: 'Fira Mono', 'Consolas', 'Menlo', monospace !important;
    }
    .main {
        background: #111 !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #00ff00 !important;
        font-family: 'Fira Mono', 'Consolas', 'Menlo', monospace !important;
        text-shadow: 0 0 10px #00ff00;
        text-align: center !important;
    }
    .catchy-title {
        color: #00ff00 !important;
        font-size: 3rem;
        font-family: 'Fira Mono', 'Consolas', 'Menlo', monospace !important;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 15px #00ff00;
        font-weight: bold;
        letter-spacing: 2px;
    }
    .subtitle {
        color: #39ff14 !important;
        font-size: 1.2rem;
        font-family: 'Fira Mono', 'Consolas', 'Menlo', monospace !important;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 0 0 5px #39ff14;
    }
    .stButton>button {
        width: 100%;
        background: #222 !important;
        color: #00ff00 !important;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.75rem;
        border: 2px solid #00ff00 !important;
        font-size: 1.1rem;
        font-family: 'Fira Mono', 'Consolas', 'Menlo', monospace !important;
        box-shadow: 0 0 10px #00ff00;
        transition: background 0.2s, color 0.2s;
    }
    .stButton>button:hover {
        background: #00ff00 !important;
        color: #111 !important;
        box-shadow: 0 0 20px #00ff00;
    }
    .stTextInput>div>div>input {
        background: #222 !important;
        color: #00ff00 !important;
        border: 1px solid #00ff00 !important;
        font-family: 'Fira Mono', 'Consolas', 'Menlo', monospace !important;
    }
    .stDateInput>div>input {
        background: #222 !important;
        color: #00ff00 !important;
        border: 1px solid #00ff00 !important;
        font-family: 'Fira Mono', 'Consolas', 'Menlo', monospace !important;
    }
    .stMetric {
        background: #222 !important;
        color: #00ff00 !important;
        border: 1px solid #00ff00 !important;
        border-radius: 8px;
        font-family: 'Fira Mono', 'Consolas', 'Menlo', monospace !important;
        box-shadow: 0 0 10px #00ff00;
    }
    .stDataFrame, .stExpander {
        background: #111 !important;
        color: #00ff00 !important;
        border: 1px solid #00ff00 !important;
        font-family: 'Fira Mono', 'Consolas', 'Menlo', monospace !important;
    }
    .stDownloadButton>button {
        background: #222 !important;
        color: #00ff00 !important;
        border: 1px solid #00ff00 !important;
        font-family: 'Fira Mono', 'Consolas', 'Menlo', monospace !important;
        box-shadow: 0 0 10px #00ff00;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="catchy-title">Terminal Chart Suite</div>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Examining OHLC movements with precision</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    ticker = st.text_input(
        "Ticker Symbol",
        value="AAPL",
        placeholder="e.g., AAPL, TSLA, MSFT"
    ).upper()

today = dt.datetime.now().date()

with col2:
    start_date = st.date_input(
        "Start Date",
        value=today - dt.timedelta(days=90),
        max_value=today
    )

with col3:
    end_date = st.date_input(
        "End Date",
        value=today,
        max_value=today
    )

if st.button("Generate Candlestick Chart"):
    if not ticker:
        st.error("Please enter a ticker symbol")
    elif start_date >= end_date:
        st.error("Start date must be before end date")
    else:
        try:
            with st.spinner(f"Fetching data for {ticker}..."):
                start_ts = pd.to_datetime(start_date)
                end_ts = pd.to_datetime(end_date)

                data = yf.download(
                    ticker,
                    start=start_ts,
                    end=end_ts,
                    progress=False,
                    auto_adjust=False
                )

                if data.empty:
                    st.error(f"No data found for ticker symbol: {ticker}")
                else:
                    data = data[['Open', 'High', 'Low', 'Close']].dropna()
                    data.index = pd.to_datetime(data.index)

                    if data.empty:
                        st.error("No valid OHLC data to plot in this date range.")
                    else:
                        st.markdown("---")
                        st.subheader(f"{ticker} Statistics")

                        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

                        current_price = float(data['Close'].iloc[-1])
                        first_close = float(data['Close'].iloc[0])
                        price_change = current_price - first_close
                        pct_change = (price_change / first_close) * 100 if first_close != 0 else 0
                        high_price = float(data['High'].max())
                        low_price = float(data['Low'].min())

                        with stat_col1:
                            st.markdown(
                                f"<span style='color:#00ff00;font-weight:bold;'>Current Price</span><br><span style='color:#00ff00;'>${current_price:.2f}</span>",
                                unsafe_allow_html=True
                            )
                        with stat_col2:
                            if price_change < 0:
                                st.markdown(
                                    f"<span style='color:red;font-weight:bold;'>Change</span><br><span style='color:red;'>${price_change:.2f} ({pct_change:.2f}%)</span>",
                                    unsafe_allow_html=True
                                )
                            else:
                                st.markdown(
                                    f"<span style='color:#00ff00;font-weight:bold;'>Change</span><br><span style='color:#00ff00;'>${price_change:.2f} ({pct_change:.2f}%)</span>",
                                    unsafe_allow_html=True
                                )
                        with stat_col3:
                            st.markdown(
                                f"<span style='color:#00ff00;font-weight:bold;'>High</span><br><span style='color:#00ff00;'>${high_price:.2f}</span>",
                                unsafe_allow_html=True
                            )
                        with stat_col4:
                            st.markdown(
                                f"<span style='color:red;font-weight:bold;'>Low</span><br><span style='color:red;'>${low_price:.2f}</span>",
                                unsafe_allow_html=True
                            )

                        st.markdown("---")

                        ohlc = data.copy()
                        ohlc['Date'] = mdates.date2num(ohlc.index)
                        ohlc = ohlc[['Date', 'Open', 'High', 'Low', 'Close']]

                        fig, ax = plt.subplots(figsize=(14, 6))
                        fig.patch.set_facecolor("#111")
                        ax.set_facecolor("#111")

                        candlestick_ohlc(
                            ax,
                            ohlc.values,
                            width=0.6,
                            colorup='#00ff00',
                            colordown='#ff1744'
                        )

                        ax.set_title(
                            f"{ticker} Share Prices ({start_date} to {end_date})",
                            color="#00ff00",
                            fontsize=18,
                            fontfamily='monospace'
                        )
                        ax.set_xlabel("Date", color="#00ff00", fontfamily='monospace')
                        ax.set_ylabel("Price (USD)", color="#00ff00", fontfamily='monospace')

                        ax.xaxis_date()
                        fig.autofmt_xdate()
                        ax.tick_params(axis='x', colors='#00ff00', labelsize=10)
                        ax.tick_params(axis='y', colors='#00ff00', labelsize=10)
                        ax.grid(True, linestyle='--', alpha=0.3, color='#00ff00')

                        st.pyplot(fig, clear_figure=True)

                        with st.expander("View Raw Data"):
                            display_data = data.round(2)
                            st.dataframe(display_data, width='stretch')

                        csv = data.to_csv()
                        st.download_button(
                            label="Download Data as CSV",
                            data=csv,
                            file_name=f"{ticker}_{start_date}_{end_date}.csv",
                            mime="text/csv"
                        )

        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            st.info("Please check the ticker symbol and try again.")