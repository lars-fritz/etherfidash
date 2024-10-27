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
    start_date = datetime.combine(start_date, datetime.min.time())
    delta_days = (start_date - reference_date).totalseconds()/(24*60*60)
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
    summary_data["Days until start date"] = delta_days
    
    yearly_data = []
    beginning_supply = current_supply

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

# Date input
start_date_str = st.date_input("Select a starting date", value=datetime(2024, 12, 1))

# Button to calculate
if st.button("Calculate Token Supply"):
    try:
        summary, yearly_data = calculate_token_supply(start_date_str)

        st.write(f"### Days until selected start date: {summary['Days until start date']}")

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
