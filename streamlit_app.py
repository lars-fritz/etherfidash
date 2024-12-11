import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from io import StringIO

# Define the URL of the CSV file
CSV_URL = "https://community.chaoslabs.xyz/aave/risk/assets/weETH"  # Replace with the direct CSV link

# Function to fetch and load the CSV data
@st.cache
def fetch_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for HTTP issues
        csv_data = StringIO(response.text)  # Convert to a file-like object
        return pd.read_csv(csv_data)  # Load CSV data into a pandas DataFrame
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error

# Fetch the data
st.title("weETH Data Visualization")
st.write("Fetching and displaying data from ChaosLabs.")

data = fetch_data(CSV_URL)

if not data.empty:
    # Display the DataFrame
    st.write("Raw Data:")
    st.dataframe(data)

    # Create a plot
    # Assuming 'Date' and 'Value' are columns in the CSV file (adjust as per actual data structure)
    if "Date" in data.columns and "Value" in data.columns:
        data["Date"] = pd.to_datetime(data["Date"])  # Convert to datetime if necessary
        fig = px.line(data, x="Date", y="Value", title="weETH Trends Over Time")
        st.plotly_chart(fig)
    else:
        st.warning("The data does not have 'Date' and 'Value' columns for plotting.")
else:
    st.warning("No data available to display.")
