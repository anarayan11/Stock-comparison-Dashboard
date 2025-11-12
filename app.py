import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="Stock Comparative Analysis Dashboard", layout="wide")

st.markdown("""
    <style>
    .stApp {background-color: #0E1117; color: #FAFAFA;}
    h1, h2, h3 {color: #00BFFF; font-weight: 800;}
    .winner-box {
        background-color: rgba(0, 191, 255, 0.15);
        border-left: 5px solid #00BFFF;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .loser-box {
        background-color: rgba(255, 127, 80, 0.15);
        border-left: 5px solid #FF7F50;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Stock Comparative Analysis Dashboard")
st.write("Compare two companies’ stock trends, financial metrics, and overall performance — with automatic insights.")

col1, col2 = st.columns(2)
with col1:
    ticker1 = st.text_input("Enter First Company Ticker:", "AAPL")
with col2:
    ticker2 = st.text_input("Enter Second Company Ticker:", "MSFT")

start_date = st.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2025-01-01"))

currency = st.selectbox("Select Currency", ["USD", "INR", "EUR"])

conversion_rates = {"USD": 1, "INR": 84.3, "EUR": 0.93}

def format_large(num):
    if isinstance(num, (int, float)):
        if num >= 1e12:
            return f"{num/1e12:.2f}T"
        elif num >= 1e9:
            return f"{num/1e9:.2f}B"
        elif num >= 1e6:
            return f"{num/1e6:.2f}M"
        else:
            return f"{num:.2f}"
    return "N/A"

@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, progress=False)
    if "Adj Close" in data.columns:
        data = data[["Adj Close"]].rename(columns={"Adj Close": "Close"})
    elif "Close" in data.columns:
        data = data[["Close"]]
    data.dropna(inplace=True)
    return data

def get_company_info(ticker):
    info = yf.Ticker(ticker).info
    return {
        "name": info.get("longName", ticker),
        "sector": info.get("sector", "N/A"),
        "marketCap": info.get("marketCap", 0),
        "peRatio": info.get("trailingPE", 0),
        "profit": info.get("grossProfits", 0),
        "revenue": info.get("totalRevenue", 0),
    }

def predict_prices(data):
    data = data.reset_index()
    data["Days"] = np.arange(len(data))
    X = data[["Days"]]
    y = data["Close"]
    model = LinearRegression()
    model.fit(X, y)
    future_days = np.arange(len(data), len(data) + 7).reshape(-1, 1)
    future_prices = model.predict(future_days)
    return float(future_prices[-1])

if st.button(" Analyze Stocks"):
    try:
        data1 = load_data(ticker1, start_date, end_date)
        data2 = load_data(ticker2, start_date, end_date)

        if data1.empty or data2.empty:
            st.error("Invalid ticker(s) or no data found.")
        else:
            combined = pd.merge(data1, data2, left_index=True, right_index=True)
            combined.columns = [f"{ticker1}_Close", f"{ticker2}_Close"]

            def stock_metrics(series):
                daily_return = series.pct_change().dropna()
                avg_price = series.mean()
                total_return = ((series.iloc[-1] - series.iloc[0]) / series.iloc[0]) * 100
                volatility = daily_return.std() * 100
                return round(avg_price, 2), round(total_return, 2), round(volatility, 2)

            avg1, ret1, vol1 = stock_metrics(combined[f'{ticker1}_Close'])
            avg2, ret2, vol2 = stock_metrics(combined[f'{ticker2}_Close'])
            pred1 = predict_prices(data1)
            pred2 = predict_prices(data2)
            info1, info2 = get_company_info(ticker1), get_company_info(ticker2)

            conv = conversion_rates[currency]
            avg1, avg2, pred1, pred2 = avg1 * conv, avg2 * conv, pred1 * conv, pred2 * conv

            st.subheader("Stock Price Comparison")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=combined.index, y=combined[f'{ticker1}_Close'] * conv,
                                     mode='lines', name=ticker1, line=dict(width=3, color='#00BFFF')))
            fig.add_trace(go.Scatter(x=combined.index, y=combined[f'{ticker2}_Close'] * conv,
                                     mode='lines', name=ticker2, line=dict(width=3, color='#FF7F50')))
            fig.update_layout(template="plotly_dark", xaxis_title="Date",
                              yaxis_title=f"Price ({currency})")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Performance Metrics")
            colA, colB = st.columns(2)

            with colA:
                st.markdown(f"### {info1['name']} ({ticker1}) — *{info1['sector']}*")
                st.markdown(f"""
                - **Average Price:** {currency} {avg1:.2f}
                - **Total Return:** {ret1}%
                - **Volatility:** {vol1}%
                - **Next 7-day Predicted Price:** {currency} {pred1:.2f}
                """)

            with colB:
                st.markdown(f"### {info2['name']} ({ticker2}) — *{info2['sector']}*")
                st.markdown(f"""
                - **Average Price:** {currency} {avg2:.2f}
                - **Total Return:** {ret2}%
                - **Volatility:** {vol2}%
                - **Next 7-day Predicted Price:** {currency} {pred2:.2f}
                """)

            st.subheader("Financial Overview")

            finance_df = pd.DataFrame({
                "Metric": ["Market Cap", "P/E Ratio", "Gross Profit", "Revenue"],
                f"{ticker1}": [
                    f"{format_large(info1['marketCap']*conv)} {currency}",
                    round(info1['peRatio'], 2),
                    f"{format_large(info1['profit']*conv)} {currency}",
                    f"{format_large(info1['revenue']*conv)} {currency}",
                ],
                f"{ticker2}": [
                    f"{format_large(info2['marketCap']*conv)} {currency}",
                    round(info2['peRatio'], 2),
                    f"{format_large(info2['profit']*conv)} {currency}",
                    f"{format_large(info2['revenue']*conv)} {currency}",
                ]
            })
            st.dataframe(finance_df)

            st.subheader("Metric Highlights")
            metrics = {
                "Total Return": ret1 > ret2,
                "Volatility": vol1 < vol2,
                "Market Cap": info1["marketCap"] > info2["marketCap"],
                "P/E Ratio": info1["peRatio"] < info2["peRatio"],
                "Revenue": info1["revenue"] > info2["revenue"],
            }

            for metric, winner in metrics.items():
                if winner:
                    st.markdown(f"<div class='winner-box'>{ticker1} outperforms in {metric}.</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='loser-box'>{ticker2} leads in {metric}.</div>", unsafe_allow_html=True)

            st.subheader("Automated Summary")
            summary = f"""
            Between **{info1['name']} ({ticker1})** and **{info2['name']} ({ticker2})**,  
            {ticker1 if ret1 > ret2 else ticker2} shows stronger stock performance with higher overall returns.  
            {ticker1 if info1['marketCap'] > info2['marketCap'] else ticker2} leads in market capitalization,  
            while {ticker1 if info1['peRatio'] < info2['peRatio'] else ticker2} offers a better valuation (lower P/E ratio).  
            In terms of revenue, {ticker1 if info1['revenue'] > info2['revenue'] else ticker2} dominates,  
            making it a more financially robust company.  
            Overall, **{ticker1 if ret1 > ret2 else ticker2}** emerges as the stronger investment candidate.
            """
            st.markdown(summary)

    except Exception as e:
        st.error(f"Error fetching data: {e}")
