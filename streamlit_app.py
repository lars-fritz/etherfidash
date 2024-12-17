import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dune_client.client import DuneClient
from sklearn.linear_model import LinearRegression

# API Setup
DUNE_API_KEY = 'Y7zXGTwljFiiIOw3MsXUqpkTf6DofTQH'  # Replace with your actual Dune API key
QUERY_ID = 4430040
dune = DuneClient(api_key=DUNE_API_KEY)

# Cache data fetching to improve performance
@st.cache_data
def fetch_data():
    """Fetches the blockchain data from Dune API."""
    return dune.get_latest_result_dataframe(query=QUERY_ID)

data = fetch_data()

# Function to process blockchain data and calculate linear regression
def process_blockchain(data, blockchain_name):
    """Processes the data for a given blockchain and computes linear regression."""
    blockchain_data = data[data["blockchain"] == blockchain_name]
    blockchain_eth_rate = blockchain_data[["day", "eth_rate"]].dropna()

    # Convert data to correct formats
    blockchain_eth_rate["eth_rate"] = pd.to_numeric(blockchain_eth_rate["eth_rate"], errors='coerce')
    blockchain_eth_rate["day"] = pd.to_datetime(blockchain_eth_rate["day"], errors='coerce')

    # Sort data chronologically to ensure positive slope (time flows forward)
    blockchain_eth_rate = blockchain_eth_rate.sort_values(by="day").reset_index(drop=True)

    # Remove invalid rows
    blockchain_eth_rate = blockchain_eth_rate.dropna().reset_index(drop=True)
    blockchain_eth_rate = blockchain_eth_rate[~blockchain_eth_rate["eth_rate"].isin([np.inf, -np.inf])]

    if blockchain_eth_rate.empty:
        st.warning(f"No valid data available for blockchain: {blockchain_name}")
        return None, None, None, None, None

    # Prepare linear regression
    X = np.arange(1, len(blockchain_eth_rate) + 1).reshape(-1, 1)
    Y = blockchain_eth_rate["eth_rate"].values.reshape(-1, 1)

    try:
        model = LinearRegression()
        model.fit(X, Y)
        predictions = model.predict(X)
        slope = model.coef_[0][0]  # Slope of the regression line
        residuals_std = np.std(Y - predictions)
        return blockchain_eth_rate["day"], Y, predictions, abs(slope), residuals_std  # Ensure positive slope
    except ValueError as e:
        st.error(f"Error fitting regression model for {blockchain_name}: {e}")
        return None, None, None, None, None

# Streamlit App Title
st.title("Interactive Blockchain ETH Rate and Weeth Liquidity Dashboard")

# Dynamically fetch blockchain names
blockchain_options = list(data["blockchain"].unique())
blockchain_options.insert(0, "All Blockchains")  # Add an "All Blockchains" option

# Blockchain selection widget
selected_blockchains = st.multiselect(
    "Select Blockchains to Visualize",
    options=blockchain_options,
    default=["scroll", "arbitrum", "ethereum"]
)

# Adjust for "All Blockchains" selection
if "All Blockchains" in selected_blockchains:
    selected_blockchains = [b for b in blockchain_options if b != "All Blockchains"]

# Define colors for blockchains dynamically
color_palette = [
    'green', 'blue', 'red', 'orange', 'purple', 'teal', 'pink', 'brown', 'cyan', 'magenta', 'grey'
]
color_map = {blockchain: color_palette[i % len(color_palette)] for i, blockchain in enumerate(blockchain_options)}

# ETH Rate Plot with Regression
st.subheader("ETH Rates with Linear Regression")
fig_eth_rate = go.Figure()

# Summary statistics dictionary
summary_stats = {}

for blockchain in selected_blockchains:
    days, Y, predictions, slope, residuals_std = process_blockchain(data, blockchain)

    if days is not None:
        # Scatter plot for actual ETH rates
        fig_eth_rate.add_trace(go.Scatter(
            x=days, y=Y.flatten(),
            mode='markers',
            marker=dict(color=color_map[blockchain]),
            name=f"{blockchain} ETH Rate"
        ))

        # Regression line
        fig_eth_rate.add_trace(go.Scatter(
            x=days, y=predictions.flatten(),
            mode='lines',
            line=dict(color=color_map[blockchain], width=2),
            name=f"{blockchain} Regression (Slope: {slope:.6f})"
        ))

        # Store summary statistics
        summary_stats[blockchain] = {
            "Slope": slope,
            "Residuals Std Dev": residuals_std
        }

# Configure the ETH Rate plot layout with legend on the right
fig_eth_rate.update_layout(
    title="ETH Rates and Linear Regression for Selected Blockchains",
    xaxis_title="Day",
    yaxis_title="ETH Rate",
    template="plotly_white",
    legend=dict(
        orientation="v",  # Vertical legend
        yanchor="top",
        y=0.9,
        xanchor="left",
        x=1.02  # Place legend to the right of the plot
    )
)
st.plotly_chart(fig_eth_rate)

# Display Summary Statistics Table
st.subheader("Summary Statistics")
summary_df = pd.DataFrame.from_dict(summary_stats, orient='index')
summary_df = summary_df.rename(columns={"Slope": "Slope (ETH Rate Change)", "Residuals Std Dev": "Std Dev of Residuals"})
st.dataframe(summary_df.style.format({"Slope (ETH Rate Change)": "{:.6f}", "Std Dev of Residuals": "{:.6f}"}))

# Footer Information
st.info("The regression line represents the trend of ETH rates over time, and the summary statistics provide insights into each blockchain's ETH rate changes.")


st.subheader("Weeth Liquidity Across Blockchains")
fig_liquidity = go.Figure()

# Process liquidity data
liquidity_data = {}

for blockchain in selected_blockchains:
    blockchain_data = data[data["blockchain"] == blockchain]
    blockchain_liquidity = blockchain_data[["day", "weeth_liquidity"]].dropna()

    blockchain_liquidity["weeth_liquidity"] = pd.to_numeric(blockchain_liquidity["weeth_liquidity"], errors='coerce')
    blockchain_liquidity["day"] = pd.to_datetime(blockchain_liquidity["day"], errors='coerce')
    blockchain_liquidity = blockchain_liquidity.sort_values(by="day").reset_index(drop=True)

    if not blockchain_liquidity.empty:
        fig_liquidity.add_trace(go.Scatter(
            x=blockchain_liquidity["day"],
            y=blockchain_liquidity["weeth_liquidity"],
            mode='markers',
            marker=dict(size=4, color=color_map[blockchain]),  # Reduced dot size here
            name=f"{blockchain} Weeth Liquidity"
        ))

# Layout for Weeth Liquidity
fig_liquidity.update_layout(
    title="Weeth Liquidity for Selected Blockchains Over Time",
    xaxis_title="Day",
    yaxis_title="Weeth Liquidity",
    legend=dict(yanchor="top", y=0.9, xanchor="left", x=1.02),
    template="plotly_white"
)
st.plotly_chart(fig_liquidity)

# Footer
st.info("Weeth Liquidity shows how much liquidity is available on each blockchain over time.")

### Normalized ETH Rates Plot ###
# Function to normalize ETH rate relative to Ethereum
def normalize_eth_rate(data, blockchain_name, ethereum_eth_rate, ethereum_days):
    blockchain_data = data[data["blockchain"] == blockchain_name]
    blockchain_eth_rate = blockchain_data[["day", "eth_rate"]].dropna()
    blockchain_eth_rate["eth_rate"] = pd.to_numeric(blockchain_eth_rate["eth_rate"], errors='coerce')
    blockchain_eth_rate["day"] = pd.to_datetime(blockchain_eth_rate["day"])
    blockchain_eth_rate = blockchain_eth_rate.sort_values(by="day").reset_index(drop=True)

    normalized_eth_rate = []
    for i, day in enumerate(blockchain_eth_rate["day"]):
        if day in ethereum_days:
            idx = ethereum_days.index(day)
            eth_rate_value = ethereum_eth_rate[idx]
            if eth_rate_value != 0:  # Avoid division by zero
                normalized_value = blockchain_eth_rate["eth_rate"].iloc[i] / eth_rate_value
                normalized_eth_rate.append(normalized_value)
            else:
                normalized_eth_rate.append(None)
        else:
            normalized_eth_rate.append(None)

    blockchain_eth_rate["normalized_eth_rate"] = normalized_eth_rate
    blockchain_eth_rate = blockchain_eth_rate.dropna(subset=["normalized_eth_rate"])
    return blockchain_eth_rate

# First Plot: ETH Rates with Linear Regression
st.subheader("ETH Rates with Linear Regression")
fig_eth_rate = go.Figure()

summary_stats = {}
for blockchain in selected_blockchains:
    days, Y, predictions, slope, residuals_std = process_blockchain(data, blockchain)
    if days is not None:
        fig_eth_rate.add_trace(go.Scatter(
            x=days, y=Y.flatten(),
            mode='markers',
            marker=dict(color=color_map[blockchain]),
            name=f"{blockchain} ETH Rate"
        ))
        fig_eth_rate.add_trace(go.Scatter(
            x=days, y=predictions.flatten(),
            mode='lines',
            line=dict(color=color_map[blockchain], width=2),
            name=f"{blockchain} Regression (Slope: {slope:.6f})"
        ))
        summary_stats[blockchain] = {"Slope": slope, "Residuals Std Dev": residuals_std}

fig_eth_rate.update_layout(
    title="ETH Rates and Linear Regression for Selected Blockchains",
    xaxis_title="Day",
    yaxis_title="ETH Rate",
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
)
st.plotly_chart(fig_eth_rate)

# Second Plot: Normalized ETH Rates Relative to Ethereum
st.subheader("Normalized ETH Rates Relative to Ethereum")
fig_normalized = go.Figure()

# Filter Ethereum data to get baseline for normalization
ethereum_data = data[data["blockchain"] == "ethereum"]
ethereum_eth_rate = ethereum_data["eth_rate"].dropna().tolist()
ethereum_days = pd.to_datetime(ethereum_data["day"].dropna()).tolist()

# Define colors for the selected blockchains
normalized_colors = ["green", "blue", "purple", "brown", "teal", "pink"]
selected_normalized_blockchains = ['scroll', 'arbitrum', 'blast', 'bnb', 'base', 'linea']

for idx, blockchain in enumerate(selected_normalized_blockchains):
    blockchain_data = normalize_eth_rate(data, blockchain, ethereum_eth_rate, ethereum_days)
    if not blockchain_data.empty:
        fig_normalized.add_trace(go.Scatter(
            x=blockchain_data["day"],
            y=blockchain_data["normalized_eth_rate"],
            mode='lines+markers',
            name=f"{blockchain.capitalize()} Relative ETH Rate",
            line=dict(color=normalized_colors[idx]),
            marker=dict(size=4)  # Reduced dot size
        ))

fig_normalized.update_layout(
    title="Normalized ETH Rates Relative to Ethereum",
    xaxis_title="Day",
    yaxis_title="Relative ETH Rate",
    legend=dict(yanchor="top", y=0.9, xanchor="left", x=1.02),
    template="plotly_white"
)
st.plotly_chart(fig_normalized)
