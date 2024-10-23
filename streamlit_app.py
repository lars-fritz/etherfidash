import streamlit as st
from datetime import datetime
import pandas as pd

# Constants
initial_token_supply = 2874597203
initial_block_rewards = 298945229
daily_reward = 230572
airdrop_tokens = 95250000
expansion_tokens = 47625000
first_two_years = 2
second_two_years = 2
inflation_rate = 1.75 / 100  # 1.75% inflation
reference_date = datetime(2024, 10, 24, 0, 0)  # October 24th 0:00 UTC

def calculate_token_supply(start_date):
    # Convert start_date to datetime
    start_date = datetime.combine(start_date, datetime.min.time())
    
    # Calculate the number of full days between the reference date and the start date
    delta_days = (start_date - reference_date).days
    if delta_days < 0:
        raise ValueError("Start date must be after or on October 24th, 2024.")
    
    # Calculate tokens added due to daily reward
    tokens_added = delta_days * daily_reward
    current_supply = initial_token_supply + tokens_added
    
    # Calculate remaining block rewards
    remaining_block_rewards = initial_block_rewards - tokens_added
    
    # Divide block rewards evenly over 4 years
    yearly_block_reward = remaining_block_rewards / 4
    
    # Initialize output
    summary_data = {}
    
    # Initial information
    summary_data["Total supply on 10/24/2024"] = int(initial_token_supply)
    summary_data["Remaining block rewards on 10/24/2024"] = int(initial_block_rewards)
    
    # Display remaining block rewards at the start date
    remaining_block_rewards_at_start = initial_block_rewards - tokens_added
    summary_data["Remaining block rewards at start date"] = int(remaining_block_rewards_at_start)
    
    # Display one quarter of the remaining block rewards
    one_quarter_remaining_rewards = remaining_block_rewards / 4
    summary_data["One quarter of the remaining block rewards"] = int(one_quarter_remaining_rewards)
    
    # Initialize yearly amounts
    summary_data["Yearly airdrop tokens"] = airdrop_tokens
    summary_data["Yearly expansion tokens"] = expansion_tokens
    
    # Inflation information
    inflation_info = (
        "Inflation kicks in after year 4, specifically in year 5.\n"
        "It is calculated as 1.75% of the total supply at the end of the previous year plus expansion tokens."
    )

    # Initialize a DataFrame for yearly data
    yearly_data = []
    
    # First two years (include airdrop tokens + expansion tokens + block rewards)
    for year in range(first_two_years):
        block_reward_tokens = min(yearly_block_reward, remaining_block_rewards)
        yearly_tokens_added = airdrop_tokens + block_reward_tokens + expansion_tokens
        current_supply += yearly_tokens_added
        remaining_block_rewards -= block_reward_tokens  # Subtract what was distributed
        
        yearly_data.append({
            "Year": year + 1,
            "Total Supply": int(current_supply),
            "Tokens Added": f"Airdrop = {airdrop_tokens}, Expansion = {expansion_tokens}, Block Rewards = {int(block_reward_tokens)}"
        })
    
    # Next two years (include only expansion tokens + block rewards)
    for year in range(second_two_years):
        block_reward_tokens = min(yearly_block_reward, remaining_block_rewards)
        yearly_tokens_added = block_reward_tokens + expansion_tokens
        current_supply += yearly_tokens_added
        remaining_block_rewards -= block_reward_tokens  # Subtract what was distributed
        
        yearly_data.append({
            "Year": year + 3,
            "Total Supply": int(current_supply),
            "Tokens Added": f"Expansion = {expansion_tokens}, Block Rewards = {int(block_reward_tokens)}"
        })
    
    # Year 5: Expansion tokens + inflation (1.75% of total supply at end of year 4 + expansion tokens)
    inflation_tokens_year_5 = inflation_rate * (current_supply + expansion_tokens)
    yearly_tokens_added = expansion_tokens + inflation_tokens_year_5
    current_supply += yearly_tokens_added
    
    yearly_data.append({
        "Year": 5,
        "Total Supply": int(current_supply),
        "Tokens Added": f"Expansion = {expansion_tokens}, Inflation = {int(inflation_tokens_year_5)}"
    })
    
    # Year 6: Expansion tokens + inflation (1.75% of total supply at end of year 5 + expansion tokens)
    inflation_tokens_year_6 = inflation_rate * (current_supply + expansion_tokens)
    yearly_tokens_added = expansion_tokens + inflation_tokens_year_6
    current_supply += yearly_tokens_added
    
    yearly_data.append({
        "Year": 6,
        "Total Supply": int(current_supply),
        "Tokens Added": f"Expansion = {expansion_tokens}, Inflation = {int(inflation_tokens_year_6)}"
    })

    # Year 7: Only inflation (1.75% of total supply at end of year 6)
    inflation_tokens_year_7 = inflation_rate * current_supply
    current_supply += inflation_tokens_year_7
    
    yearly_data.append({
        "Year": 7,
        "Total Supply": int(current_supply),
        "Tokens Added": f"Inflation = {int(inflation_tokens_year_7)}"
    })
    
    return summary_data, inflation_info, yearly_data

# Streamlit app layout
st.title("Token Supply Calculator")

# Date input
start_date_str = st.date_input("Select a starting date", value=datetime(2024, 12, 1))

# Button to calculate
if st.button("Calculate Token Supply"):
    try:
        summary, inflation_info, yearly_data = calculate_token_supply(start_date_str)
        
        # Display summary in a nice box
        st.markdown("### Initial Information")
        st.markdown(
            f"""
            <div style="border: 2px solid #1E90FF; padding: 10px; border-radius: 5px;">
            <ul>
                {''.join(f'<li>{key}: {value}</li>' for key, value in summary.items())}
            </ul>
            </div>
            """, unsafe_allow_html=True
        )
        
        st.markdown("### Inflation Information")
        st.write(inflation_info)

        # Convert yearly data to a DataFrame and display as a table
        yearly_df = pd.DataFrame(yearly_data)
        st.markdown("### Yearly Data")
        st.table(yearly_df)
        
    except ValueError as e:
        st.error(str(e))
