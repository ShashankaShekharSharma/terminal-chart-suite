# Terminal Chart Suite

Terminal Chart Suite is a focused financial visualization tool designed to examine OHLC price movements with clarity and precision.  
The application provides a structured environment for traders, analysts, and researchers who rely on candlestick behavior and price-action patterns to interpret market sentiment and understand trend development.

---

## Overview

This project offers a direct and disciplined way to analyze market data using traditional candlestick charting. By visualizing the true relationship between opens, highs, lows, and closes across any chosen timeframe, users can better understand:

- Shifts in market sentiment  
- Strength or weakness of directional moves  
- Volatility conditions  
- Reversal and continuation structures  
- Historical price reactions at key levels  

The interface intentionally adopts a terminal-style theme to maintain analytical focus and remove unnecessary distractions.

---

## Features

### Candlestick Charting
The application renders accurate OHLC candlestick charts using unadjusted market data, ensuring that each candle reflects true historical price action rather than adjusted pricing.

### Financial Metrics
For each selected date window, the application calculates and displays:

- Current closing price  
- Net change over the selected period  
- Percentage change  
- Highest price reached  
- Lowest price reached  

These values provide a quick summary of performance and volatility.

### Custom Date Range Selection
Users may define any historical window. The system automatically handles weekends, holidays, and missing data.

### Raw Data Access
All OHLC values used in the chart are viewable within the application and downloadable as a CSV file for further study or reporting.

### Finance-Centric Terminal Interface
The dark, monochrome layout keeps attention on structure, movement, and market rhythm rather than interface elements, making it suitable for focused chart review.

---

## How to Use

1. Enter a valid ticker symbol.  
2. Select a start and end date for the period you want to examine.  
3. Click “Generate Candlestick Chart” to load the data and visualize price movements.  
4. Review the displayed metrics to understand performance over the chosen period.  
5. Explore the candlestick chart to identify structural patterns or changes in trend behavior.  
6. Open the raw data panel if you need to inspect or export the underlying OHLC records.

The application retrieves market data from Yahoo Finance and uses Matplotlib’s candlestick plotting tools to display price action.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/<repository-name>.git
cd terminal-chart-suite
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```
