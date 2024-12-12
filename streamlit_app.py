import streamlit as st
import requests
from bs4 import BeautifulSoup

st.title("ETH Balance from Etherscan")

try:
    # URL of the Etherscan page
    url = "https://etherscan.io/address/0x308861a430be4cce5502d0a12724771fc6daf216"
    
    st.write("Fetching data from:", url)  # Debug log
    
    # Send a GET request
    response = requests.get(url)
    st.write("Response Status Code:", response.status_code)  # Debug log
    
    # Parse the HTML
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Locate the ETH balance
    balance_div = soup.find("h4", string="ETH Balance").find_next_sibling("div")
    
    if balance_div:
        eth_balance = balance_div.text.strip()
        st.write("ETH Balance:", eth_balance)
    else:
        st.error("Failed to locate ETH balance on the page.")
except Exception as e:
    st.error(f"An error occurred: {e}")
