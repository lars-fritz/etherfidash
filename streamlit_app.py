# Install required libraries (run these in your terminal before running the app)
# !pip install streamlit dune_client dataclasses_json matplotlib plotly scikit-learn pandas numpy

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dune_client.client import DuneClient
from sklearn.linear_model import LinearRegression

# API Setup
DUNE_API_KEY = 'Y7zXGTwljFiiIOw3MsXUqpkTf6DofTQH'
QUERY_ID = 4430040
dune = DuneClient(api_key=DUNE_API_KEY)

# Fetch data from Dune API
@st.cache_data
def fetch_data():
    return dune.get_latest_result_dataframe(query=QUERY_ID)

data = fetch_data()

# Data processing functions
def process_blockchain(data, blockchain_name):
    blockchain_data = data[data["blockchain"] == blockchain_name]
    blockchain_eth_rate = blockchain_data[["day", "eth_rate"]].dropna()
    blockchain_eth_rate["eth_rate"] = pd.to_numeric(blockchain_eth_rate["eth_rate"], errors='coerce')
    blockchain_eth_rate["day"] = pd.to_datetime(blockchain_eth_rate["day"])
    blockchain_eth_rate = blockchain_eth_rate.sort_values(by="day").reset_index(drop=True)

    if blockchain_eth_rate.empty:
        return None, None, None, None

    X = np.arange(1, len(blockchain_eth_rate) + 1).reshape(-1, 1)
    Y = blockchain_eth_rate["eth_rate"].values.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, Y)
    predictions = model.predict(X)
    slope = model.coef_[0][0]
    return blockchain_eth_rate["day"], Y, predictions, slope

# Streamlit Page Title
st.title("Interactive Blockchain ETH Rate and Weeth Liquidity Dashboard")

# Blockchain Selection
selected_blockchains = st.multiselect("Select Blockchains to Visualize",
                                      options=data["blockchain"].unique(),
                                      default=["scroll", "arbitrum", "ethereum"])

# Plot ETH Rates
st.subheader("ETH Rates with Linear Regression")
fig_eth_rate = go.Figure()
for blockchain in selected_blockchains:
    days, Y, predictions, slope = process_blockchain(data, blockchain)
    if days is not None:
        fig_eth_rate.add_trace(go.Scatter(x=days, y=Y.flatten(),
                                          mode='markers', name=f"{blockchain} ETH Rate"))
        fig_eth_rate.add_trace(go.Scatter(x=days, y=predictions.flatten(),
                                          mode='lines', name=f"{blockchain} Regression (Slope: {slope:.4f})"))

fig_eth_rate.update_layout(
    title="ETH Rates for Selected Blockchains",
    xaxis_title="Day",
    yaxis_title="ETH Rate",
    template="plotly_white"
)
st.plotly_chart(fig_eth_rate)

# Weeth Liquidity Plot
st.subheader("Weeth Liquidity Over Time")
fig_liquidity = go.Figure()
for blockchain in selected_blockchains:
    blockchain_data = data[data["blockchain"] == blockchain]
    if not blockchain_data.empty:
        blockchain_data_clean = blockchain_data[["day", "weeth_liquidity"]].dropna()
        blockchain_data_clean["day"] = pd.to_datetime(blockchain_data_clean["day"])
        fig_liquidity.add_trace(go.Scatter(x=blockchain_data_clean["day"],
                                           y=blockchain_data_clean["weeth_liquidity"],
                                           mode='lines+markers', name=f"{blockchain} Liquidity"))

fig_liquidity.update_layout(
    title="Weeth Liquidity for Selected Blockchains",
    xaxis_title="Day",
    yaxis_title="Weeth Liquidity",
    template="plotly_white"
)
st.plotly_chart(fig_liquidity)

# Summary
st.subheader("Summary Statistics")
summary = {}
for blockchain in selected_blockchains:
    _, Y, predictions, _ = process_blockchain(data, blockchain)
    if Y is not None:
        residuals_std = np.std(Y - predictions)
        summary[blockchain] = f"Standard Deviation of Residuals: {residuals_std:.4f}"

for blockchain, stat in summary.items():
    st.write(f"**{blockchain}**: {stat}")

