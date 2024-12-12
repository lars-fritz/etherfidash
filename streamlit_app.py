import requests
from bs4 import BeautifulSoup

# URL of the Etherscan page
url = "https://etherscan.io/address/0x308861a430be4cce5502d0a12724771fc6daf216#code"

# Send a GET request to the page
response = requests.get(url)

# Parse the HTML content
soup = BeautifulSoup(response.text, 'html.parser')

# Locate the ETH Balance (adjust the element ID if necessary)
try:
    eth_balance_element = soup.find("span", {"id": "ContentPlaceHolder1_divSummary"}).find("b")
    eth_balance = eth_balance_element.text.strip()  # Extract the text and remove any surrounding whitespace
    print(f"ETH Balance: {eth_balance}")
except AttributeError:
    print("Failed to locate ETH balance on the page.")
