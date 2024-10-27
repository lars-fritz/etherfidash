import streamlit as st
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import time

# Constants
initial_block_rewards = 3175000000
daily_average_reward = 2000
daily_average_burn = 300
airdrop_tokens = 95250000
expansion_tokens = 47625000
first_two_years = 2
second_two_years = 2
inflation_rate = 1.75 / 100  # 1.75% inflation
reference_date = datetime.now()  # Set the reference date to the current datetime

# Selenium setup to retrieve data
def retrieve_initial_values():
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(options=options)

    # Retrieve initial token supply
    driver.get("https://explorer.fantom.network/staking")
    try:
        initial_token_supply_element = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.XPATH, "//span[contains(text(),'Total Supply')]/following-sibling::span"))
        )
        initial_token_supply = int(initial_token_supply_element.text.replace(",", ""))
    except TimeoutException:
        initial_token_supply = 2872024300  # Fallback value
        st.warning("Could not retrieve initial token supply. Using fallback value.")

    # Retrieve initial burn
    driver.get("https://ftmonfire.fantom.network")
    try:
        initial_burn_element = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.XPATH, "//element-path-for-burn"))
        )
        initial_burn = int(initial_burn_element.text.replace(",", ""))
    except TimeoutException:
        initial_burn = 11483890  # Fallback value
        st.warning("Could not retrieve initial burn. Using fallback value.")
        
    driver.quit()
    return initial_token_supply, initial_burn

initial_token_supply, initial_burn = retrieve_initial_values()
initial_block_rewards -= initial_burn + initial_token_supply

# Define the token supply calculation function
def calculate_token_supply(start_date):
    # Convert start_date to datetime
    start_date = datetime.combine(start_date, datetime.min.time())
    
    # Calculate the number of full days between the reference date and the start date
    delta_days = (start_date - reference_date).days
    if delta_days < 0:
        raise ValueError("Start date must be after or on the reference date.")
    
    # Calculate tokens added due to daily reward
    tokens_added = delta_days * daily_average_reward
    tokens_removed = delta_days * daily_average_burn
    current_supply = initial_token_supply + tokens_added - tokens_removed
    
    # Calculate remaining block rewards
    remaining_block_rewards = initial_block_rewards - tokens_added
    
    # Divide block rewards evenly over 4 years
    yearly_block_reward = remaining_block_rewards / 4
    
    # Initialize output
    summary_data = {}
    summary_data["Total supply on reference date"] = int(initial_token_supply)
    summary_data["Remaining block rewards on reference date"] = int(initial_block_rewards)
    summary_data["Daily block rewards"] = int(daily_average_reward)
    summary_data["Remaining block rewards at start date"] = int(initial_block_rewards - tokens_added)
    summary_data["Remaining yearly block rewards"] = int(yearly_block_reward)
    
    # Initialize a DataFrame for yearly data
    yearly_data = []
    beginning_supply = current_supply
    
    # Add data for years
    for year in range(7):
        if year < 2:
            yearly_tokens_added = airdrop_tokens + yearly_block_reward + expansion_tokens
        elif year < 4:
            yearly_tokens_added = yearly_block_reward + expansion_tokens
        elif year == 4:
            inflation_tokens = inflation_rate * beginning_supply
            yearly_tokens_added = expansion_tokens + inflation_tokens
        elif year == 5:
            inflation_tokens = inflation_rate * (beginning_supply + expansion_tokens)
            yearly_tokens_added = expansion_tokens + inflation_tokens
        else:
            inflation_tokens = inflation_rate * beginning_supply
            yearly_tokens_added = inflation_tokens
        
        end_of_year_supply = beginning_supply + yearly_tokens_added
        
        yearly_data.append({
            "Year": year + 1,
            "Beginning of Year Supply": int(beginning_supply),
            "End of Year Supply": int(end_of_year_supply),
            "Airdrop Tokens": airdrop_tokens if year < 2 else 0,
            "Expansion Tokens": expansion_tokens if year < 6 else 0,
            "Inflation Tokens": int(inflation_tokens) if year >= 4 else 0,
            "Block Rewards": int(yearly_block_reward if year < 4 else 0),
        })
        
        beginning_supply = end_of_year_supply
        remaining_block_rewards -= yearly_block_reward
    
    return summary_data, yearly_data

# Streamlit app layout
st.title("Token Supply Calculator")

# Display summary information before the date input
st.markdown("### Initial Information")
st.markdown(
    f"""
    <div style="border: 2px solid #1E90FF; padding: 10px; border-radius: 5px;">
    <ul>
        <li>Total supply on reference date: {initial_token_supply}</li>
        <li>Remaining block rewards on reference date: {initial_block_rewards}</li>
        <li>Daily average block rewards: {daily_average_reward}</li>
        <li>Yearly airdrop tokens: {airdrop_tokens}</li>
        <li>Yearly expansion tokens: {expansion_tokens}</li>
        <li>Inflation kicks in after year 4, starting at 1.75% of the total supply in year 5.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True
)

# Date input
start_date_str = st.date_input("Select a starting date", value=datetime(2024, 12, 1))

# Button to calculate
if st.button("Calculate Token Supply"):
    try:
        summary, yearly_data = calculate_token_supply(start_date_str)

        # Convert yearly data to a DataFrame
        yearly_df = pd.DataFrame(yearly_data)
        st.markdown("### Yearly Data")
        st.table(yearly_df)
        
        # Plot token supply over time starting from year 0
        plt.figure(figsize=(10, 6))
        plt.plot(yearly_df["Year"], yearly_df["End of Year Supply"], marker='o', color='b', label="End of Year Supply")
        plt.xlabel("Year")
        plt.ylabel("Token Supply")
        plt.title("Token Supply Over Time")
        plt.grid(True)
        plt.legend()
        
        # Display the plot in Streamlit
        st.pyplot(plt)
        
    except ValueError as e:
        st.error(str(e))
