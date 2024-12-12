import requests
from bs4 import BeautifulSoup

# URL of the Etherscan page
url = "https://etherscan.io/address/0x308861a430be4cce5502d0a12724771fc6daf216"

# Send a GET request to fetch the page content
response = requests.get(url)

# Parse the HTML content
soup = BeautifulSoup(response.text, 'html.parser')

# Locate the ETH balance
try:
    # Find the specific div containing the ETH balance
    balance_div = soup.find("h4", string="ETH Balance").find_next_sibling("div").text.strip()
    print(f"ETH Balance: {balance_div}")
except AttributeError:
    print("Failed to locate ETH balance on the page.")
