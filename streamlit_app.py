import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
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

columns_to_remove = [
    "index", "date", "liquidity_usd", "total_usd", 
    "total_eth", "weeth_liquidity_usd", "total_volume", 
    "total_rate", "utilization"
]

# Create the reduced dataset
reduced_data = (
    data.head(10)               # Take the first 10 rows
    .drop(index=3, errors="ignore")  # Drop the fourth row (index starts at 0)
    .drop(columns=columns_to_remove, errors="ignore")  # Drop specified columns
)

# Convert eth_rate column to numeric (coerce non-numeric values to NaN)
reduced_data["eth_rate"] = pd.to_numeric(reduced_data["eth_rate"], errors="coerce")

# Drop rows where eth_rate is NaN (if necessary)
reduced_data = reduced_data.dropna(subset=["eth_rate"])

# Get the reference_eth_rate from row 0
reference_eth_rate = reduced_data.iloc[0]["eth_rate"]

# Ensure reference_eth_rate is numeric
if pd.isna(reference_eth_rate):
    raise ValueError("Reference eth_rate (row 0) is NaN or invalid.")

# Add the relative_deviation column
reduced_data["relative_deviation"] = (
    (reduced_data["eth_rate"] - reference_eth_rate) / reference_eth_rate
)



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
st.title("Weeth Dashboard")

# Dynamically fetch blockchain names
blockchain_options = list(data["blockchain"].unique())
blockchain_options.insert(0, "All Blockchains")  # Add an "All Blockchains" option

# Blockchain selection widget
selected_blockchains = st.multiselect(
    "Select Blockchains to Visualize",
    options=blockchain_options,
    default=["scroll", "arbitrum", "ethereum","blast","linea","bnb","base","optimism"]
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
st.subheader("ETH Rates across blockchains including some statistics")
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
st.subheader("Linear regression")
summary_df = pd.DataFrame.from_dict(summary_stats, orient='index')
summary_df = summary_df.rename(columns={"Slope": "Slope (ETH Rate Change)", "Residuals Std Dev": "Std Dev of Residuals"})
st.dataframe(summary_df.style.format({"Slope (ETH Rate Change)": "{:.6f}", "Std Dev of Residuals": "{:.6f}"}))

residuals_std_dict = {}

# Process each selected blockchain and collect residuals_std
for blockchain in selected_blockchains:
    days, Y, predictions, slope, residuals_std = process_blockchain(data, blockchain)
    
    if residuals_std is not None:
        residuals_std_dict[blockchain] = residuals_std

# Add the standard deviation column to the reduced data
def add_std_deviation_column(reduced_data, residuals_std_dict):
    # Map blockchain to its standard deviation of residuals
    reduced_data["std_deviation"] = reduced_data["blockchain"].map(residuals_std_dict)
    return reduced_data

# Update reduced_data to include the new column
reduced_data = add_std_deviation_column(reduced_data, residuals_std_dict)



# Footer Information
st.info("The regression line represents the trend of ETH rates over time whereas the standard deviation is a measure for the volatility on the same time window (although it is unclear how to extend volatility to pegged tokens)")

### Relative ETH Rate Difference Plot ###
st.subheader("Relative Deviation from Z-All")

# Initialize a new interactive plot
fig_relative_diff = go.Figure()

# Preprocess Z-All baseline data
zall_data = data[data["blockchain"] == "Z-All"]
zall_days = pd.to_datetime(zall_data["day"].dropna()).tolist()
zall_eth_rate = pd.to_numeric(zall_data["eth_rate"].dropna(), errors='coerce').tolist()

# Function to calculate relative ETH rate difference
def calculate_relative_difference(data, blockchain_name, zall_eth_rate, zall_days):
    blockchain_data = data[data["blockchain"] == blockchain_name]
    blockchain_eth_rate = blockchain_data[["day", "eth_rate"]].dropna()
    blockchain_eth_rate["eth_rate"] = pd.to_numeric(blockchain_eth_rate["eth_rate"], errors='coerce')
    blockchain_eth_rate["day"] = pd.to_datetime(blockchain_eth_rate["day"])
    blockchain_eth_rate = blockchain_eth_rate.sort_values(by="day").reset_index(drop=True)

    relative_difference = []
    for i, day in enumerate(blockchain_eth_rate["day"]):
        if day in zall_days:
            idx = zall_days.index(day)
            eth_rate_zall = zall_eth_rate[idx]
            if eth_rate_zall != 0:  # Avoid division by zero
                diff = (blockchain_eth_rate["eth_rate"].iloc[i] - eth_rate_zall) / eth_rate_zall
                relative_difference.append(diff)
            else:
                relative_difference.append(None)
        else:
            relative_difference.append(None)

    blockchain_eth_rate["relative_difference"] = relative_difference
    blockchain_eth_rate = blockchain_eth_rate.dropna(subset=["relative_difference"])
    return blockchain_eth_rate

# List of blockchains to process: show only selected ones
relative_blockchains = [b for b in selected_blockchains if b != "Z-All"]

# Default colors for blockchains
relative_diff_colors = {
    'scroll': 'red', 'arbitrum': 'blue', 'blast': 'green', 'bnb': 'purple',
    'base': 'orange', 'linea': 'pink', 'optimism': 'teal', 'ethereum': 'gray'
}

# Table data for averages and standard deviations
table_data = []

# Plot relative differences for each selected blockchain
for blockchain in relative_blockchains:
    blockchain_relative = calculate_relative_difference(data, blockchain, zall_eth_rate, zall_days)
    if not blockchain_relative.empty:
        # Add plot for each blockchain's relative difference
        fig_relative_diff.add_trace(go.Scatter(
            x=blockchain_relative["day"],
            y=blockchain_relative["relative_difference"],
            mode='lines+markers',
            name=f"{blockchain.capitalize()} Relative ETH Rate Difference",
            line=dict(color=relative_diff_colors.get(blockchain, "gray")),  # Default to gray if no color
            marker=dict(size=4)  # Smaller marker size
        ))

        # Calculate average and standard deviation of the relative differences for the blockchain
        avg_diff = blockchain_relative["relative_difference"].mean()
        std_diff = blockchain_relative["relative_difference"].std()

        # Append the results to the table data
        table_data.append({
            "Blockchain": blockchain.capitalize(),
            "Average Relative Difference": round(avg_diff, 4),
            "Standard Deviation": round(std_diff, 4)
        })

# Update layout for the plot
fig_relative_diff.update_layout(
    title="Relative ETH Rate Difference Compared to Z-All",
    xaxis_title="Day",
    yaxis_title="Relative Difference",
    legend=dict(yanchor="top", y=0.9, xanchor="left", x=1.02),
    template="plotly_white"
)

# Display the plot in Streamlit
st.plotly_chart(fig_relative_diff)

# Display the table below the plot with average and standard deviation for each blockchain
st.subheader("Average and Standard Deviation for Each Blockchain")
table_df = pd.DataFrame(table_data)

# Show the table in Streamlit
st.table(table_df)

# Footer Information
st.info("Relative differences are now calculated using Z-All as the baseline for comparison. It is important to note that weETH can be staked/minted on ethereum, linea, blast, and base. Curiously, they do not seem to agree in value across the three chains for 1 ETH staked. ")

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


st.subheader("Liquidity Across Blockchains")
fig_liquidity = go.Figure()

# Process liquidity data
liquidity_data = {}

for blockchain in selected_blockchains:
    blockchain_data = data[data["blockchain"] == blockchain]
    blockchain_liquidity = blockchain_data[["day", "liquidity_eth"]].dropna()

    blockchain_liquidity["liquidity_eth"] = pd.to_numeric(blockchain_liquidity["liquidity_eth"], errors='coerce')
    blockchain_liquidity["day"] = pd.to_datetime(blockchain_liquidity["day"], errors='coerce')
    blockchain_liquidity = blockchain_liquidity.sort_values(by="day").reset_index(drop=True)

    if not blockchain_liquidity.empty:
        fig_liquidity.add_trace(go.Scatter(
            x=blockchain_liquidity["day"],
            y=blockchain_liquidity["liquidity_eth"],
            mode='markers',
            marker=dict(size=4, color=color_map[blockchain]),  # Reduced dot size here
            name=f"{blockchain} Liquidity (ETH)"
        ))

# Layout for Liquidity in ETH
fig_liquidity.update_layout(
    title="Liquidity (ETH) for Selected Blockchains Over Time",
    xaxis_title="Day",
    yaxis_title="Liquidity (ETH)",
    legend=dict(yanchor="top", y=0.9, xanchor="left", x=1.02),
    template="plotly_white"
)
st.plotly_chart(fig_liquidity)

# Footer
st.info("Liquidity (ETH) shows how much liquidity is available on each blockchain over time.")

# Step 1: Calculate daily eth_price_day more robustly
def calculate_eth_price_day(data):
    eth_price_day = {}
    
    for day in data["day"].unique():
        daily_data = data[data["day"] == day]
        z_all_row = daily_data[daily_data["blockchain"] == "Z-All"]
        
        if not z_all_row.empty:
            # Convert to numeric and handle NaN values
            liquidity_usd = pd.to_numeric(z_all_row["liquidity_usd"].iloc[0], errors='coerce')
            liquidity_eth = pd.to_numeric(z_all_row["liquidity_eth"].iloc[0], errors='coerce')
            
            # Only calculate price if both values are valid and liquidity_eth is positive
            if pd.notna(liquidity_usd) and pd.notna(liquidity_eth) and liquidity_eth > 0:
                eth_price_day[day] = liquidity_usd / liquidity_eth
            else:
                eth_price_day[day] = None
        else:
            eth_price_day[day] = None
    
    return eth_price_day

# Step 2: Calculate volume in ETH safely
def calculate_volume_eth(data):
    # Make a copy to avoid modifying the original dataframe
    df = data.copy()
    
    # Calculate ETH prices for each day
    eth_price_day = calculate_eth_price_day(df)
    
    # Convert eth_price_day dictionary to a series mapped to the day column
    df["eth_price_day"] = df["day"].map(eth_price_day)
    
    # Convert volume to numeric, replacing invalid values with NaN
    df["volume"] = pd.to_numeric(df["volume"], errors='coerce')
    
    # Calculate volume_eth only where we have valid values
    df["volume_eth"] = df.apply(
        lambda row: row["volume"] / row["eth_price_day"] 
        if pd.notna(row["volume"]) and pd.notna(row["eth_price_day"]) and row["eth_price_day"] != 0 
        else None,
        axis=1
    )
    
    # Step 3: Calculate liquidity-to-volume ratio
    df["liquidity_adjusted"] = df.apply(
        lambda row: (
            pd.to_numeric(row["liquidity_eth"], errors='coerce') +
            (pd.to_numeric(row["weeth_liquidity"], errors='coerce') * 
             pd.to_numeric(row["eth_rate"], errors='coerce'))
        ),
        axis=1
    )
    
    df["liquidity_to_volume_ratio"] = df.apply(
        lambda row: (
            row["liquidity_adjusted"] / row["volume_eth"]
            if pd.notna(row["liquidity_adjusted"]) and 
               pd.notna(row["volume_eth"]) and 
               row["volume_eth"] != 0
            else None
        ),
        axis=1
    )
    
    return df

# Apply the calculations
data = calculate_volume_eth(data)

# Step 4: Create the plot
st.subheader("Liquidity-to-Volume Ratio Across Blockchains")
fig_ratio = go.Figure()

for blockchain in selected_blockchains:
    blockchain_data = data[data["blockchain"] == blockchain]
    blockchain_data = blockchain_data[["day", "liquidity_to_volume_ratio"]].dropna()

    blockchain_data["day"] = pd.to_datetime(blockchain_data["day"], errors="coerce")
    blockchain_data = blockchain_data.sort_values(by="day").reset_index(drop=True)

    if not blockchain_data.empty:
        fig_ratio.add_trace(go.Scatter(
            x=blockchain_data["day"],
            y=blockchain_data["liquidity_to_volume_ratio"],
            mode='markers',
            marker=dict(size=4, color=color_map[blockchain]),  # Reduced dot size here
            name=f"{blockchain} Liquidity/Volume Ratio"
        ))

# Layout for Liquidity-to-Volume Ratio Plot
fig_ratio.update_layout(
    title="Liquidity-to-Volume Ratio for Selected Blockchains Over Time",
    xaxis_title="Day",
    yaxis_title="Liquidity-to-Volume Ratio",
    legend=dict(yanchor="top", y=0.9, xanchor="left", x=1.02),
    template="plotly_white"
)

st.plotly_chart(fig_ratio)

# Footer
st.info("Liquidity-to-Volume Ratio indicates the efficiency of liquidity utilization across blockchains.")



# Title of the Streamlit app
# Title of the Streamlit app
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests  # For fetching data from CoinGecko API


# Title of the Streamlit app
st.title("Collateral Analysis and CSV Upload")

# Function to get the current price of ETH from CoinGecko
def get_eth_price():
    try:
        url = "https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd"
        
        # Add headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            
            # Explicitly check for 'ethereum' key
            if 'ethereum' in data and 'usd' in data['ethereum']:
                return data['ethereum']['usd']
            else:
                st.error("Unexpected response format from CoinGecko API")
                return None
        elif response.status_code == 429:
            st.warning("Rate limit exceeded. Using a fallback method.")
            return None
        else:
            st.error(f"Failed to fetch ETH price. Status code: {response.status_code}")
            return None
    
    except requests.RequestException as e:
        st.error(f"Network error fetching ETH price: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error fetching ETH price: {e}")
        return None

# Fallback method to get ETH price
def get_eth_price_fallback():
    # You can add multiple fallback sources here
    fallback_sources = [
        "https://api.binance.com/api/v3/ticker/price?symbol=ETHUSDT",
        # Add more API endpoints as needed
    ]
    
    for source in fallback_sources:
        try:
            response = requests.get(source, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                # Handle different API response formats
                if source == "https://api.binance.com/api/v3/ticker/price?symbol=ETHUSDT":
                    return float(data['price'])
                
                # Add more source-specific parsing as needed
        except Exception as e:
            st.error(f"Error fetching from {source}: {e}")
    
    return None

# Fetch the current ETH price
eth_price = get_eth_price()

# Fallback to alternative method if primary method fails
if eth_price is None:
    eth_price = get_eth_price_fallback()

# Handle ETH price fetching
if eth_price is not None:
    st.write(f"The current price of Ethereum (ETH) is: ${eth_price} USD")
else:
    st.warning("Could not fetch the current ETH price. Using default value of 0.")
    eth_price = 3500

# Mapping to normalize the labels
label_mapping = {
    'Arbitrum-main': 'arbitrum',
    'Base-main': 'base',
    'Ethereum-main': 'ethereum',
    'Scroll-main': 'scroll'
}

# Step 1: Upload the Total Collateral in Dollar CSV Data
st.subheader("Upload Total Collateral in Dollar Data")
uploaded_collateral_file = st.file_uploader("Upload a CSV file for Total Collateral (in Dollar)", type=["csv"])

# Check if the file is uploaded
if uploaded_collateral_file is not None:
    # Step 2: Store the uploaded Total Collateral data
    collateral_data = pd.read_csv(uploaded_collateral_file)
    
    # Display the first few rows of the uploaded data for user verification
    st.write("Preview of the Total Collateral in Dollar Data:")

    # Replace the labels using the mapping dictionary
    collateral_data['Label'] = collateral_data['Label'].replace(label_mapping)

    # Convert collateral values to ETH if ETH price is available
    if eth_price > 0:
        collateral_data['Value (ETH)'] = collateral_data['Value'] / eth_price
    else:
        collateral_data['Value (ETH)'] = None

    # Show the preview table with the ETH column
    st.write(collateral_data[['Label', 'Value', 'Value (ETH)']])

    # Store the collateral data in session state for later use
    st.session_state.collateral_data = collateral_data

    # Create two columns for displaying pie charts
    col1, col2 = st.columns(2)

    # Generate a pie chart for Total Collateral in Dollar (USD)
    with col1:
        fig_collateral_usd = go.Figure(go.Pie(
            labels=collateral_data['Label'],
            values=collateral_data['Value'],
            hole=0.3,
            title="Total Collateral in Dollar",
            textinfo="label+percent+value",  # Show label, percent and value
            hoverinfo="label+value+percent"  # Show label, value, and percent on hover
        ))
        st.plotly_chart(fig_collateral_usd)

    # Generate a pie chart for Total Collateral in ETH
    with col2:
        fig_collateral_eth = go.Figure(go.Pie(
            labels=collateral_data['Label'],
            values=collateral_data['Value (ETH)'],
            hole=0.3,
            title="Total Collateral in ETH",
            textinfo="label+percent+value",  # Show label, percent and value
            hoverinfo="label+value+percent"  # Show label, value, and percent on hover
        ))
        st.plotly_chart(fig_collateral_eth)

else:
    st.write("No file uploaded for Total Collateral. Please upload a CSV file.")

# Step 3: Upload the Collateral at Risk CSV Data
st.subheader("Upload Collateral at Risk Data")
uploaded_risk_file = st.file_uploader("Upload a CSV file for Collateral at Risk", type=["csv"])

# Check if the file is uploaded
if uploaded_risk_file is not None:
    # Step 4: Store the uploaded Collateral at Risk data
    risk_data = pd.read_csv(uploaded_risk_file)
    
    # Display the first few rows of the uploaded data for user verification
    st.write("Preview of the Collateral at Risk Data:")

    # Replace the labels using the mapping dictionary
    risk_data['Label'] = risk_data['Label'].replace(label_mapping)

    # Convert collateral risk values to ETH if ETH price is available
    if eth_price > 0:
        risk_data['Value (ETH)'] = risk_data['Value'] / eth_price
    else:
        risk_data['Value (ETH)'] = None

    # Show the preview table with the ETH column
    st.write(risk_data[['Label', 'Value', 'Value (ETH)']])

    # Store the collateral risk data in session state for later use
    st.session_state.risk_data = risk_data

    # Create two columns for displaying pie charts for Collateral at Risk
    col1, col2 = st.columns(2)

    # Generate a pie chart for Collateral at Risk in Dollar (USD)
    with col1:
        fig_risk_usd = go.Figure(go.Pie(
            labels=risk_data['Label'],
            values=risk_data['Value'],
            hole=0.3,
            title="Collateral at Risk",
            textinfo="label+percent+value",  # Show label, percent and value
            hoverinfo="label+value+percent"  # Show label, value, and percent on hover
        ))
        st.plotly_chart(fig_risk_usd)

    # Generate a pie chart for Collateral at Risk in ETH
    with col2:
        fig_risk_eth = go.Figure(go.Pie(
            labels=risk_data['Label'],
            values=risk_data['Value (ETH)'],
            hole=0.3,
            title="Collateral at Risk in ETH",
            textinfo="label+percent+value",  # Show label, percent and value
            hoverinfo="label+value+percent"  # Show label, value, and percent on hover
        ))
        st.plotly_chart(fig_risk_eth)
else:
    st.write("No file uploaded for Collateral at Risk. Please upload a CSV file.")
def add_collateral_at_risk(reduced_data, risk_data):
    # Create a mapping from Label to Value (ETH) in risk_data
    risk_mapping = dict(zip(risk_data['Label'], risk_data['Value (ETH)']))
        
    # Map the risk data to the reduced dataset based on blockchain
    reduced_data["collateral_at_risk_eth"] = reduced_data["blockchain"].map(risk_mapping)
    return reduced_data

    # Update reduced_data with collateral at risk column
reduced_data = add_collateral_at_risk(reduced_data, risk_data)
if eth_price > 0:
    # Handle invalid or missing values in the "volume" column
    if "volume" in reduced_data.columns:
        reduced_data["volume"] = pd.to_numeric(reduced_data["volume"], errors="coerce")  # Convert to numeric
        reduced_data["volume"] = reduced_data["volume"].fillna(0)  # Replace NaN with 0
        
        # Perform the division and rename the column
        reduced_data["volume"] = reduced_data["volume"] / eth_price
        reduced_data = reduced_data.rename(columns={"volume": "volume_eth"})
        

    else:
        st.error("The 'volume' column is missing from the reduced dataset.")
else:
    st.error("ETH price is not available or invalid. Unable to convert volume to ETH.")
st.write("Reduced Dataset with Collateral at Risk (in ETH):")
st.table(reduced_data)
# Prepare summary data
summary_rows = []

def safe_get_latest_rate(data, blockchain):
    """
    Safely extract the latest rate for a given blockchain with extensive error handling.
    
    Args:
        data (pd.DataFrame): The input DataFrame containing blockchain data
        blockchain (str): The name of the blockchain to get rate for
        
    Returns:
        float or None: The latest rate for the blockchain, or None if unavailable
    """
    try:
        # Filter for the specific blockchain
        blockchain_data = data[data["blockchain"] == blockchain].copy()
        
        if blockchain_data.empty:
            print(f"No data found for blockchain: {blockchain}")
            return None
            
        # Convert day column to datetime if it exists
        if "day" in blockchain_data.columns:
            blockchain_data['day'] = pd.to_datetime(blockchain_data['day'], errors='coerce')
            
            # Remove rows with invalid dates
            blockchain_data = blockchain_data.dropna(subset=['day'])
            
            if blockchain_data.empty:
                print(f"No valid dates found for blockchain: {blockchain}")
                return None
                
            # Get the latest date
            latest_date = blockchain_data['day'].max()
            latest_data = blockchain_data[blockchain_data['day'] == latest_date]
        else:
            print(f"No 'day' column found for blockchain: {blockchain}")
            return None
            
        # Extract eth_rate values
        if "eth_rate" not in latest_data.columns:
            print(f"No 'eth_rate' column found for blockchain: {blockchain}")
            return None
            
        # Convert eth_rate to numeric, handling any non-numeric values
        latest_data['eth_rate'] = pd.to_numeric(latest_data['eth_rate'], errors='coerce')
        eth_rates = latest_data['eth_rate'].dropna()
        
        if eth_rates.empty:
            print(f"No valid eth_rate values found for blockchain: {blockchain}")
            return None
            
        # Return the first valid rate
        return float(eth_rates.iloc[0])
        
    except Exception as e:
        print(f"Error processing {blockchain}: {str(e)}")
        return None

# Safely calculate relative difference
def safe_calculate_relative_difference(latest_rate, baseline_rate):
    if latest_rate is not None and baseline_rate is not None and baseline_rate != 0:
        try:
            return (latest_rate - baseline_rate) / baseline_rate
        except (TypeError, ZeroDivisionError):
            return None
    return None

# Get Ethereum's latest rate
ethereum_latest_rate = safe_get_latest_rate(data, "ethereum")

for blockchain in selected_blockchains:
    # Skip Ethereum in comparisons
    if blockchain == "ethereum":
        continue
    
    # Get latest ETH rate for the blockchain
    latest_eth_rate = safe_get_latest_rate(data, blockchain)
    
    # Calculate relative difference
    relative_difference = safe_calculate_relative_difference(latest_eth_rate, ethereum_latest_rate)
    
    # Find standard deviation from previous relative difference calculation
    std_dev = None
    if 'table_data' in locals():
        for row in table_data:
            if row['Blockchain'].lower() == blockchain:
                std_dev = row['Standard Deviation']
                break
    
    # Liquidity (eth_liquidity and weeth_liquidity)
    blockchain_data = data[data["blockchain"] == blockchain]
    blockchain_data['day'] = pd.to_datetime(blockchain_data['day'])
    latest_data = blockchain_data.loc[blockchain_data['day'] == blockchain_data['day'].max()]
    
    eth_liquidity_data = latest_data["liquidity_eth"].dropna()
    weeth_liquidity_data = latest_data["weeth_liquidity"].dropna()
    
    eth_liquidity = float(eth_liquidity_data.iloc[0]) if not eth_liquidity_data.empty else None
    weeth_liquidity = float(weeth_liquidity_data.iloc[0]) if not weeth_liquidity_data.empty else None
    
    # Collateral at Risk (if available in session state)
    collateral_at_risk_eth = None
    if hasattr(st.session_state, 'risk_data'):
        risk_row = st.session_state.risk_data[st.session_state.risk_data['Label'] == blockchain]
        if not risk_row.empty:
            collateral_at_risk_eth = float(risk_row['Value (ETH)'].values[0])
    
    # Calculate Collateral at Risk / Liquidity Ratio
    risk_liquidity_ratio = None
    if collateral_at_risk_eth is not None and eth_liquidity is not None and eth_liquidity != 0:
        risk_liquidity_ratio = eth_liquidity / collateral_at_risk_eth
    
    # Determine if minting is possible
    can_mint = 'Yes' if blockchain in ['ethereum', 'blast', 'base', 'linea'] else 'No'
    
    # Prepare row for summary
    summary_row = {
        'Blockchain': blockchain.capitalize(),
        'Latest ETH Rate': round(latest_eth_rate, 6) if latest_eth_rate is not None else 'N/A',
        'Relative ETH Rate Difference': round(relative_difference, 4) if relative_difference is not None else 'N/A',
        'Std Dev of Relative Difference': round(std_dev, 4) if std_dev is not None else 'N/A',
        'ETH Liquidity': round(eth_liquidity, 2) if eth_liquidity is not None else 'N/A',
        'Weeth Liquidity': round(weeth_liquidity, 2) if weeth_liquidity is not None else 'N/A',
        'Collateral at Risk (ETH)': round(collateral_at_risk_eth, 2) if collateral_at_risk_eth is not None else 'N/A',
        'Liquidity/CollateralAtRisk': round(risk_liquidity_ratio, 4) if risk_liquidity_ratio is not None else 'N/A',
        'Can Mint': can_mint
    }
    
    summary_rows.append(summary_row)

# Create DataFrame from summary rows
summary_df = pd.DataFrame(summary_rows)

# Display the summary table
st.table(summary_df)
st.info("Everything is calculated relative to Z-All now which is not the same as the minting rate. We might want to use that instead.")
