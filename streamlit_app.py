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

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# --- Title and description ---
st.title("Weeth Liquidity Across Blockchains")
st.write("This app visualizes **Weeth Liquidity** across various blockchains over time.")

# --- List of blockchains and other settings ---
blockchains = ['scroll', 'arbitrum', 'ethereum', 'optimism', 'blast', 'base', 'linea', 'bnb']
colors = ['green', 'blue', 'red', 'orange', 'purple', 'teal', 'pink', 'brown']
labels = ['Scroll', 'Arbitrum', 'Ethereum', 'Optimism', 'Blast', 'Base', 'Linea', 'BNB']

# --- File upload ---
uploaded_file = st.file_uploader("Upload your query results (CSV file)", type=["csv"])

# --- Process uploaded file ---
if uploaded_file is not None:
    # Read the uploaded CSV file
    query_result = pd.read_csv(uploaded_file)

    # Create a dictionary to hold liquidity data for each blockchain
    liquidity_data = {}

    # Process each blockchain
    for blockchain in blockchains:
        # Filter for rows matching the blockchain
        blockchain_data = query_result[query_result["blockchain"] == blockchain]

        # Extract and clean 'weeth_liquidity' and 'day'
        blockchain_data_clean = blockchain_data[["day", "weeth_liquidity"]].dropna()
        blockchain_data_clean["weeth_liquidity"] = pd.to_numeric(blockchain_data_clean["weeth_liquidity"], errors='coerce')
        blockchain_data_clean["day"] = pd.to_datetime(blockchain_data_clean["day"])
        blockchain_data_clean = blockchain_data_clean.sort_values(by="day").reset_index(drop=True)

        # Store cleaned liquidity data
        liquidity_data[blockchain] = blockchain_data_clean["weeth_liquidity"].tolist()

    # --- Plotting ---
    st.subheader("Weeth Liquidity Over Time")

    # Create a Matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Scatter plot for each blockchain
    for idx, blockchain in enumerate(blockchains):
        if len(liquidity_data[blockchain]) > 0:
            ax.scatter(range(1, len(liquidity_data[blockchain]) + 1), liquidity_data[blockchain],
                       color=colors[idx], s=10, label=f"{labels[idx]} Weeth Liquidity")

    # Add labels, grid, and legend
    ax.set_xlabel("Days (Counted from 1)")
    ax.set_ylabel("Weeth Liquidity")
    ax.set_title("Weeth Liquidity for Blockchains Over Time")
    ax.grid(which='major', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend()

    # Adjust layout
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)

else:
    st.warning("Please upload a CSV file to proceed.")
