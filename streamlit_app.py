import streamlit as st
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

# Constants and user-defined inputs
initial_block_rewards = 3175000000
daily_average_reward = 2000
daily_average_burn = 300
airdrop_tokens = 95250000
expansion_tokens = 47625000
first_two_years = 2
second_two_years = 2
inflation_rate = 1.75 / 100  # 1.75% inflation

# User input for current token supply and current burn
st.title("Token Supply Calculator")

initial_token_supply = st.number_input(
    "Enter the current token supply:",
    min_value=0,
    value=2872024300,
    help="You can find the current token supply at https://explorer.fantom.network/staking."
)
initial_burn = st.number_input(
    "Enter the current burn:",
    min_value=0,
    value=11483890,
    help="You can find the current burn at https://ftmonfire.fantom.network."
)

reference_date = datetime.now()  # Set reference date to current date and time
initial_block_rewards -= initial_burn + initial_token_supply

# Define the token supply calculation function
def calculate_token_supply(start_date):
    # Calculate precise time difference in days with decimals
    delta_days = (start_date - reference_date).total_seconds() / (24 * 3600)
    if delta_days < 0:
        raise ValueError("Start date must be after or on the reference date.")
    
    tokens_added = delta_days * daily_average_reward
    tokens_removed = delta_days * daily_average_burn
    current_supply = initial_token_supply + tokens_added - tokens_removed
    
    remaining_block_rewards = initial_block_rewards - tokens_added
    yearly_block_reward = remaining_block_rewards / 4
    
    summary_data = {}
    summary_data["Total supply on reference date"] = int(initial_token_supply)
    summary_data["Remaining block rewards on reference date"] = int(initial_block_rewards)
    summary_data["Daily block rewards"] = int(daily_average_reward)
    summary_data["Remaining block rewards at start date"] = int(initial_block_rewards - tokens_added)
    summary_data["Remaining yearly block rewards"] = int(yearly_block_reward)
    summary_data["Days until start date"] = round(delta_days, 2)  # Days as decimal

    # Rest of the yearly calculations remain unchanged...
    # Define the yearly data calculations...

    return summary_data, yearly_data

# Date input
start_date_str = st.date_input("Select a starting date", value=datetime(2024, 12, 1))

# Button to calculate
if st.button("Calculate Token Supply"):
    try:
        summary, yearly_data = calculate_token_supply(start_date_str)

        st.write(f"### Days until selected start date: {summary['Days until start date']} days")

        yearly_df = pd.DataFrame(yearly_data)
        st.markdown("### Yearly Data")
        st.table(yearly_df)
        
        plt.figure(figsize=(10, 6))
        plt.plot(yearly_df["Year"], yearly_df["End of Year Supply"], marker='o', color='b', label="End of Year Supply")
        plt.xlabel("Year")
        plt.ylabel("Token Supply")
        plt.title("Token Supply Over Time")
        plt.grid(True)
        plt.legend()
        
        st.pyplot(plt)
        
    except ValueError as e:
        st.error(str(e))

