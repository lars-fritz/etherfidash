from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import streamlit as st

# Streamlit app title
st.title("ETH Balance from Etherscan")

try:
    # Configure Selenium to use headless mode (no browser window)
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Runs Chrome in headless mode
    chrome_options.add_argument("--disable-gpu")  # Disable GPU for headless environments
    chrome_options.add_argument("--no-sandbox")
    
    # Path to ChromeDriver (replace with your actual path to chromedriver)
    service = Service('/path/to/chromedriver')  # Replace with the path to your chromedriver
    driver = webdriver.Chrome(service=service, options=chrome_options)

    # Open the Etherscan page
    url = "https://etherscan.io/address/0x308861a430be4cce5502d0a12724771fc6daf216"
    driver.get(url)
    
    # Wait for the page to load (adjust sleep time if necessary)
    time.sleep(5)  # Allow time for the page to load completely

    # Locate the ETH balance element by its structure
    balance_element = driver.find_element(By.XPATH, "//h4[contains(text(), 'ETH Balance')]/following-sibling::div")
    eth_balance = balance_element.text

    # Display the ETH balance on the Streamlit app
    st.write("ETH Balance:", eth_balance)

except Exception as e:
    st.error(f"An error occurred: {e}")
finally:
    driver.quit()
