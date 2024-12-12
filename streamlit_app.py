import requests
from bs4 import BeautifulSoup
import streamlit as st

st.title("ETH Balance from Etherscan")

# URL of the Etherscan page
url = "https://etherscan.io/address/0x308861a430be4cce5502d0a12724771fc6daf216"

try:
    # Send a GET request to fetch the page content
    response = requests.get(url)
    
    # Check the response
    st.write("Response Status Code:", response.status_code)
    
    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')

    # Debug: Print the first 500 characters of the HTML to check the structure
    st.write(soup.prettify()[:500])

    # Locate the ETH balance (debugging step)
    balance_div = soup.find("h4", string="ETH Balance")
    if balance_div:
        balance_element = balance_div.find_next_sibling("div")
        if balance_element:
            eth_balance = balance_element.text.strip()
            st.write("ETH Balance:", eth_balance)
        else:
            st.error("Could not find the ETH balance sibling div.")
    else:
        st.error("Could not find the 'ETH Balance' heading.")

except Exception as e:
    st.error(f"An error occurred: {e}")
