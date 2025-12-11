"""
Terminal Chart Suite — Portfolio Candles (Multi-Stock Support)
- Generates candlestick charts for all portfolio stocks or single ticker
- Ring chart is shown in its own column; candlesticks appear only after pressing Generate
"""

import io
import streamlit as st
import datetime as dt
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_finance import candlestick_ohlc

st.set_page_config(page_title="Terminal Chart Suite — Portfolio", layout="wide")

st.markdown("""
    <style>
    body, .stApp { background: #050505 !important; color: #00ff00 !important; font-family: 'Fira Mono', 'Consolas', 'Menlo', monospace !important; }
    .main { background: #050505 !important; }
    .catchy-title { color: #00ff00 !important; font-size: 2.4rem; text-align: center; margin-bottom: 0.25rem; text-shadow: 0 0 12px #00ff00; letter-spacing: 1px; font-weight:700; }
    .subtitle { color: #39ff14 !important; font-size: 0.95rem; text-align: center; margin-bottom: 1rem; text-shadow: 0 0 6px #39ff14; }
    .stButton>button, .stDownloadButton>button { background: #0b0b0b !important; color: #00ff00 !important; border: 1px solid #00ff00 !important; font-family: 'Fira Mono', monospace !important; }
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stDateInput>div>input, .stSelectbox>div>div>select { background: #0b0b0b !important; color: #00ff00 !important; border: 1px solid #00ff00 !important; font-family: 'Fira Mono', monospace !important; }
    .stDataFrame, .stExpander { background: #050505 !important; color: #00ff00 !important; border: 1px solid #00ff00 !important; }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="catchy-title">Terminal Chart Suite</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Portfolio allocation and candlestick charts — add your holdings to begin</div>', unsafe_allow_html=True)

if "portfolio_df" not in st.session_state:
    st.session_state.portfolio_df = pd.DataFrame(columns=["ticker", "shares"])

def prepare_ohlc_values(df_ohlc):
    df = df_ohlc.copy()
    required = ["Open", "High", "Low", "Close"]
    for col in required:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            raise ValueError(f"Missing column: {col}")
    df = df.dropna(subset=required)
    if df.empty:
        return np.empty((0, 5))
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return np.empty((0, 5))
    date_array = df.index.to_pydatetime()
    if not isinstance(date_array, (np.ndarray, list, tuple)):
        date_array = np.array([date_array])
    if len(date_array) == 0:
        return np.empty((0, 5))
    date_nums = mdates.date2num(date_array)
    vals = np.column_stack([date_nums, df["Open"].values, df["High"].values, df["Low"].values, df["Close"].values])
    if vals.ndim == 1:
        vals = vals.reshape(1, -1)
    return vals

def create_candlestick_chart(ticker, data, start_d, end_d):
    # Reuse your existing logic for stats + plot
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    current_price = float(data["Close"].iloc[-1])
    first_close = float(data["Close"].iloc[0])
    price_change = current_price - first_close
    pct_change = (price_change / first_close) * 100 if first_close != 0 else 0
    high_price = float(data["High"].max())
    low_price = float(data["Low"].min())

    with stat_col1:
        st.markdown(f"<span style='color:#00ff00;font-weight:bold;'>Current Price</span><br><span style='color:#00ff00;'>${current_price:.2f}</span>", unsafe_allow_html=True)
    with stat_col2:
        color = "red" if price_change < 0 else "#00ff00"
        st.markdown(f"<span style='color:{color};font-weight:bold;'>Change</span><br><span style='color:{color};'>${price_change:.2f} ({pct_change:.2f}%)</span>", unsafe_allow_html=True)
    with stat_col3:
        st.markdown(f"<span style='color:#00ff00;font-weight:bold;'>High</span><br><span style='color:#00ff00;'>${high_price:.2f}</span>", unsafe_allow_html=True)
    with stat_col4:
        st.markdown(f"<span style='color:red;font-weight:bold;'>Low</span><br><span style='color:red;'>${low_price:.2f}</span>", unsafe_allow_html=True)

    ohlc = data.copy()
    ohlc["Date"] = mdates.date2num(ohlc.index.to_pydatetime())
    ohlc_vals = ohlc[["Date", "Open", "High", "Low", "Close"]].values.astype(float)
    ohlc_plot = [tuple(row) for row in ohlc_vals] if isinstance(ohlc_vals, np.ndarray) else list(ohlc_vals)

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#111")
    ax.set_facecolor("#111")
    candlestick_ohlc(ax, ohlc_plot, width=0.6, colorup="#00ff00", colordown="#ff1744")
    ax.set_title(f"{ticker} Share Prices ({start_d} to {end_d})", color="#00ff00", fontsize=18, fontfamily="monospace")
    ax.set_xlabel("Date", color="#00ff00", fontfamily="monospace")
    ax.set_ylabel("Price (USD)", color="#00ff00", fontfamily="monospace")
    ax.xaxis_date()
    fig.autofmt_xdate()
    ax.tick_params(axis="x", colors="#00ff00", labelsize=10)
    ax.tick_params(axis="y", colors="#00ff00", labelsize=10)
    ax.grid(True, linestyle="--", alpha=0.3, color="#00ff00")
    st.pyplot(fig, clear_figure=True)

left_col, right_col = st.columns([1.2, 2])

with left_col:
    st.subheader("Portfolio input")
    st.markdown("Add holdings using the quick-add form, edit the portfolio table (if supported), or upload a CSV with columns `ticker,shares`.\n\nCSV must have headers `ticker` and `shares` (case-insensitive).")

    uploaded = st.file_uploader("Upload CSV (columns: ticker, shares)", type=["csv"])
    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded)
            df_up.columns = [c.lower() for c in df_up.columns]
            if {"ticker", "shares"}.issubset(set(df_up.columns)):
                df_up = df_up[["ticker", "shares"]].copy()
                df_up["ticker"] = df_up["ticker"].astype(str).str.upper().str.strip()
                df_up["shares"] = pd.to_numeric(df_up["shares"], errors="coerce").fillna(0)
                st.session_state.portfolio_df = df_up.reset_index(drop=True)
                st.success("CSV loaded into portfolio.")
            else:
                st.error("CSV must contain columns named 'ticker' and 'shares' (case-insensitive).")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

    with st.form("quick_add_form", clear_on_submit=True):
        c1, c2 = st.columns([2, 1])
        with c1:
            new_ticker = st.text_input("Ticker symbol to add")
        with c2:
            new_shares = st.number_input("Shares", min_value=0.0, value=0.0, step=1.0)
        add_clicked = st.form_submit_button("Add")
        if add_clicked:
            t = (new_ticker or "").strip().upper()
            if not t:
                st.error("Enter a ticker symbol to add.")
            else:
                new_row = pd.DataFrame([{"ticker": t, "shares": float(new_shares)}])
                st.session_state.portfolio_df = pd.concat([st.session_state.portfolio_df, new_row], ignore_index=True)
                st.success(f"Added {t} to portfolio.")

    st.markdown("---")
    st.markdown("Edit portfolio below (if supported) or use the textarea to load entries manually.")

    if hasattr(st, "data_editor"):
        try:
            edited = st.data_editor(st.session_state.portfolio_df, num_rows="dynamic", use_container_width=True)
            if {"ticker", "shares"}.issubset(set([c.lower() for c in edited.columns])):
                edited.columns = [c.lower() for c in edited.columns]
                edited["ticker"] = edited["ticker"].astype(str).str.upper().str.strip()
                edited["shares"] = pd.to_numeric(edited["shares"], errors="coerce").fillna(0)
                st.session_state.portfolio_df = edited[["ticker", "shares"]]
        except Exception:
            st.info("Table editor exists but failed; use the textarea or CSV upload.")
            st.dataframe(st.session_state.portfolio_df, use_container_width=True)
    else:
        st.markdown("Editable table not available in this Streamlit build. Use the textarea below or upload CSV.")
        def df_to_text(df):
            return "\n".join(f"{r.ticker},{r.shares}" for r in df.itertuples(index=False))
        portfolio_text = st.text_area("Holdings (one per line: TICKER, SHARES)", value=df_to_text(st.session_state.portfolio_df), height=160)
        if st.button("Load portfolio from textarea"):
            rows = []
            for line in portfolio_text.splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")] if "," in line else line.split()
                if len(parts) < 2:
                    continue
                ticker = parts[0].upper()
                try:
                    shares = float(parts[1])
                except Exception:
                    shares = 0.0
                rows.append({"ticker": ticker, "shares": shares})
            if rows:
                st.session_state.portfolio_df = pd.DataFrame(rows)
                st.success("Portfolio loaded from textarea.")
            else:
                st.error("No valid rows found in textarea.")

    st.markdown("---")
    st.subheader("Chart settings")
    today = dt.datetime.now().date()
    start_date = st.date_input("Start date (candles)", value=today - dt.timedelta(days=90), max_value=today)
    end_date = st.date_input("End date (candles)", value=today, max_value=today)
    compute = st.button("Compute portfolio")

def render_single_ticker(ticker_viz, s_date_viz, e_date_viz, output_container):
    try:
        data = yf.download(
            ticker_viz,
            start=pd.to_datetime(s_date_viz),
            end=pd.to_datetime(e_date_viz),
            progress=False,
            auto_adjust=False,
        )

        if data.empty:
            output_container.error(f"No data found for ticker symbol: {ticker_viz}")
            return

        data = data[["Open", "High", "Low", "Close"]].dropna()
        data.index = pd.to_datetime(data.index)
        if data.empty:
            output_container.error("No valid OHLC data to plot in this date range.")
            return

        # Render inside container
        with output_container:
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            current_price = float(data["Close"].iloc[-1])
            first_close = float(data["Close"].iloc[0])
            price_change = current_price - first_close
            pct_change = (price_change / first_close) * 100 if first_close != 0 else 0
            high_price = float(data["High"].max())
            low_price = float(data["Low"].min())

            with stat_col1:
                st.markdown(
                    f"<span style='color:#00ff00;font-weight:bold;'>Current Price</span><br><span style='color:#00ff00;'>${current_price:.2f}</span>",
                    unsafe_allow_html=True,
                )
            with stat_col2:
                if price_change < 0:
                    st.markdown(
                        f"<span style='color:red;font-weight:bold;'>Change</span><br><span style='color:red;'>${price_change:.2f} ({pct_change:.2f}%)</span>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"<span style='color:#00ff00;font-weight:bold;'>Change</span><br><span style='color:#00ff00;'>${price_change:.2f} ({pct_change:.2f}%)</span>",
                        unsafe_allow_html=True,
                    )
            with stat_col3:
                st.markdown(
                    f"<span style='color:#00ff00;font-weight:bold;'>High</span><br><span style='color:#00ff00;'>${high_price:.2f}</span>",
                    unsafe_allow_html=True,
                )
            with stat_col4:
                st.markdown(
                    f"<span style='color:red;font-weight:bold;'>Low</span><br><span style='color:red;'>${low_price:.2f}</span>",
                    unsafe_allow_html=True,
                )

            ohlc = data.copy()
            ohlc["Date"] = mdates.date2num(ohlc.index.to_pydatetime())
            ohlc_vals = ohlc[["Date", "Open", "High", "Low", "Close"]].values.astype(float)
            if isinstance(ohlc_vals, np.ndarray):
                ohlc_plot = [tuple(row) for row in ohlc_vals]
            else:
                ohlc_plot = list(ohlc_vals)

            fig, ax = plt.subplots(figsize=(14, 6))
            fig.patch.set_facecolor("#111")
            ax.set_facecolor("#111")

            candlestick_ohlc(ax, ohlc_plot, width=0.6, colorup="#00ff00", colordown="#ff1744")

            ax.set_title(
                f"{ticker_viz} Share Prices ({s_date_viz} to {e_date_viz})",
                color="#00ff00",
                fontsize=18,
                fontfamily="monospace",
            )
            ax.set_xlabel("Date", color="#00ff00", fontfamily="monospace")
            ax.set_ylabel("Price (USD)", color="#00ff00", fontfamily="monospace")

            ax.xaxis_date()
            fig.autofmt_xdate()
            ax.tick_params(axis="x", colors="#00ff00", labelsize=10)
            ax.tick_params(axis="y", colors="#00ff00", labelsize=10)
            ax.grid(True, linestyle="--", alpha=0.3, color="#00ff00")

            st.pyplot(fig, clear_figure=True)

            with st.expander("View Raw Data"):
                display_data = data.round(2)
                st.dataframe(display_data, width="stretch")

            csv = data.to_csv()
            st.download_button(
                label="Download Data as CSV",
                data=csv,
                file_name=f"{ticker_viz}_{s_date_viz}_{e_date_viz}.csv",
                mime="text/csv",
            )

    except Exception as e:
        output_container.error(f"Error fetching data: {str(e)}")
        output_container.info("Please check the ticker symbol and try again.")

with right_col:
    st.subheader("Portfolio output")
    # We'll show allocation and valuation after compute. Candlesticks rendering will be placed into a separate container
    allocation_placeholder = st.empty()
    # this placeholder will be replaced/filled by the Generate button (initially empty)
    candles_placeholder = st.empty()
    valuation_placeholder = st.empty()

    st.markdown("### Stock Visualisation")
    portfolio_tickers = []
    if not st.session_state.portfolio_df.empty:
        portfolio_df_viz = st.session_state.portfolio_df.copy()
        portfolio_df_viz["ticker"] = portfolio_df_viz["ticker"].astype(str).str.upper().str.strip()
        portfolio_df_viz["shares"] = pd.to_numeric(portfolio_df_viz["shares"], errors="coerce").fillna(0)
        portfolio_df_viz = portfolio_df_viz[portfolio_df_viz["shares"] > 0]
        portfolio_tickers = portfolio_df_viz["ticker"].unique().tolist()
    
    vcol1, vcol2, vcol3, vcol4 = st.columns([2, 1.5, 1.5, 1])
    with vcol1:
        viz_mode = st.radio("Visualisation mode", ["All Portfolio Stocks", "Single Ticker"], horizontal=True, key="viz_mode")
    with vcol2:
        s_date_viz = st.date_input("Start Date", value=today - dt.timedelta(days=90), max_value=today, key="viz_start")
    with vcol3:
        e_date_viz = st.date_input("End Date", value=today, max_value=today, key="viz_end")
    
    if viz_mode == "Single Ticker":
        with vcol4:
            ticker_viz = st.text_input("Ticker", value="AAPL", key="viz_ticker").upper()
    else:
        ticker_viz = None

    # the button that triggers candlestick drawing (candles are not shown until this is pressed)
    if st.button("Generate Candlestick Chart(s)", key="viz_generate"):
        if s_date_viz >= e_date_viz:
            st.error("Start date must be before end date")
        elif viz_mode == "Single Ticker":
            if not ticker_viz:
                st.error("Please enter a ticker symbol")
            else:
                # render single ticker into the candles placeholder
                try:
                    with st.spinner(f"Fetching data for {ticker_viz}..."):
                        # clear and use placeholder for rendering
                        candles_placeholder.empty()
                        render_single_ticker(ticker_viz, s_date_viz, e_date_viz, candles_placeholder)
                except Exception as e:
                    st.error(f"Error fetching data: {str(e)}")
                    st.info("Please check the ticker symbol and try again.")
        else:
            # All portfolio stocks -> render into the candles placeholder
            if not portfolio_tickers:
                st.error("Your portfolio is empty. Add stocks to your portfolio first.")
            else:
                candles_placeholder.empty()
                with candles_placeholder.container():
                    st.markdown(f"### Generating charts for {len(portfolio_tickers)} portfolio stocks...")
                    tabs = st.tabs(portfolio_tickers)
                    for tab_obj, ticker in zip(tabs, portfolio_tickers):
                        with tab_obj:
                            try:
                                with st.spinner(f"Fetching data for {ticker}..."):
                                    data = yf.download(ticker, start=pd.to_datetime(s_date_viz), end=pd.to_datetime(e_date_viz), progress=False, auto_adjust=False)
                                    if data.empty:
                                        st.error(f"No data found for ticker symbol: {ticker}")
                                        continue
                                    data = data[["Open", "High", "Low", "Close"]].dropna()
                                    data.index = pd.to_datetime(data.index)
                                    if data.empty:
                                        st.error(f"No valid OHLC data to plot for {ticker} in this date range.")
                                        continue
                                    create_candlestick_chart(ticker, data, s_date_viz, e_date_viz)
                                    with st.expander(f"View {ticker} Raw Data"):
                                        st.dataframe(data.round(2), use_container_width=True)
                                    csv = data.to_csv()
                                    st.download_button(f"Download {ticker} Data as CSV", data=csv, file_name=f"{ticker}_{s_date_viz}_{e_date_viz}.csv", mime="text/csv", key=f"csv_{ticker}")
                            except Exception as e:
                                st.error(f"Error fetching data for {ticker}: {str(e)}")

# Compute button: compute allocation & valuation and place the ring chart in its own column.
if compute:
    portfolio_df = st.session_state.get("portfolio_df", pd.DataFrame(columns=["ticker", "shares"])).copy()
    portfolio_df["shares"] = pd.to_numeric(portfolio_df["shares"], errors="coerce").fillna(0)
    portfolio_df["ticker"] = portfolio_df["ticker"].astype(str).str.upper().str.strip()
    portfolio_df = portfolio_df[portfolio_df["shares"] > 0].reset_index(drop=True)

    if portfolio_df.empty:
        st.error("Portfolio is empty. Add holdings before computing.")
    else:
        tickers = portfolio_df["ticker"].unique().tolist()
        prices = {}
        failed_price = []
        try:
            batch = yf.download(tickers=" ".join(tickers), period="7d", group_by="ticker", progress=False, auto_adjust=False)
            if isinstance(batch.columns, pd.MultiIndex):
                for t in tickers:
                    try:
                        close_series = batch[t]["Close"].dropna()
                        prices[t] = float(close_series.iloc[-1])
                    except Exception:
                        failed_price.append(t)
            else:
                close_series = batch["Close"].dropna()
                prices[tickers[0]] = float(close_series.iloc[-1])
        except Exception:
            for t in tickers:
                try:
                    tmp = yf.download(t, period="7d", progress=False, auto_adjust=False)
                    close_series = tmp["Close"].dropna()
                    prices[t] = float(close_series.iloc[-1])
                except Exception:
                    failed_price.append(t)

        if failed_price:
            st.warning(f"Could not fetch recent price for: {', '.join(failed_price)}. These will be omitted from valuation.")

        rows = []
        for r in portfolio_df.itertuples(index=False):
            t = r.ticker
            s = r.shares
            price = prices.get(t, np.nan)
            mv = price * s if not np.isnan(price) else np.nan
            rows.append({"ticker": t, "shares": s, "price": price, "market_value": mv})
        val_df = pd.DataFrame(rows).dropna(subset=["market_value"])

        if val_df.empty:
            st.error("No market values available for holdings. Check ticker symbols or try again later.")
        else:
            val_df["pct"] = (val_df["market_value"] / val_df["market_value"].sum()) * 100

            # Layout: left column shows allocation ring chart + valuation table; right column reserved for candlesticks (initially empty)
            alloc_col, cand_col = st.columns([1, 2])

            # Render allocation in alloc_col
            with alloc_col:
                fig_alloc, ax_alloc = plt.subplots(figsize=(5, 5), facecolor="#050505")
                ax_alloc.set_facecolor("#050505")
                labels = val_df["ticker"].tolist()
                sizes = val_df["market_value"].tolist()
                base_greens = plt.get_cmap("Greens")
                base_purples = plt.get_cmap("Purples")
                colors = []
                n = len(labels)
                for i in range(n):
                    if i % 2 == 0:
                        colors.append(base_greens(0.6 + 0.4 * (i / max(1, n - 1))))
                    else:
                        colors.append(base_purples(0.6 + 0.4 * (i / max(1, n - 1))))
                wedges, texts = ax_alloc.pie(sizes, labels=None, startangle=90, counterclock=False, wedgeprops=dict(width=0.36, edgecolor="#050505"), colors=colors)
                centre = plt.Circle((0, 0), 0.64, fc="#050505")
                ax_alloc.add_artist(centre)
                legend_labels = [f"{t} — ${mv:,.2f} ({pct:.1f}%)" for t, mv, pct in zip(val_df["ticker"], val_df["market_value"], val_df["pct"])]
                ax_alloc.legend(wedges, legend_labels, title="Allocation", loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, prop={"family": "monospace", "size": 9}, labelcolor="#00ff00")
                ax_alloc.text(0, 0, f"Total\n${val_df['market_value'].sum():,.2f}", ha="center", va="center", color="#00ff00", fontsize=10, family="monospace")
                ax_alloc.axis("equal")
                # display in this column
                allocation_placeholder.pyplot(fig_alloc, clear_figure=True)

                # Valuation table under the chart
                with valuation_placeholder:
                    st.markdown("### Valuation")
                    display = val_df[["ticker", "shares", "price", "market_value", "pct"]].copy()
                    display.columns = ["Ticker", "Shares", "Price", "Market Value", "Allocation (%)"]
                    display["Price"] = display["Price"].map(lambda x: f"${x:,.2f}")
                    display["Market Value"] = display["Market Value"].map(lambda x: f"${x:,.2f}")
                    display["Allocation (%)"] = display["Allocation (%)"].map(lambda x: f"{x:.2f}%")
                    st.dataframe(display, use_container_width=True)
                    csv_buf = val_df.to_csv(index=False)
                    st.download_button("Download valuation CSV", data=csv_buf, file_name="portfolio_valuation.csv", mime="text/csv")
                    buf_alloc = io.BytesIO()
                    fig_alloc.savefig(buf_alloc, format="png", dpi=200, bbox_inches="tight", facecolor=fig_alloc.get_facecolor())
                    buf_alloc.seek(0)
                    st.download_button("Download allocation chart (PNG)", data=buf_alloc.getvalue(), file_name="allocation_chart.png", mime="image/png")
                    if failed_price:
                        st.markdown("Warning: could not retrieve prices for the following tickers (omitted from valuation):")
                        st.write(", ".join(failed_price))

            # Reserve the right column for candlesticks. Do NOT auto-populate them here.
            with cand_col:
                # Clear any previous content and show an instructional message
                candles_placeholder.empty()
                with candles_placeholder.container():
                    st.markdown("### Candlestick charts (press 'Generate Candlestick Chart(s)' to load here)")
                    st.info("Candlestick charts will appear in this column when you press the Generate button above.")

# End of script
