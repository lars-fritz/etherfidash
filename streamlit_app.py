import streamlit as st
from datetime import datetime

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
    output = []
    
    # First two years (include airdrop tokens + expansion tokens + block rewards)
    for year in range(first_two_years):
        block_reward_tokens = min(yearly_block_reward, remaining_block_rewards)
        yearly_tokens_added = airdrop_tokens + block_reward_tokens + expansion_tokens
        current_supply += yearly_tokens_added
        remaining_block_rewards -= block_reward_tokens  # Subtract what was distributed
        
        output.append(
            f"End of year {year + 1}:\n"
            f"  Total supply = {int(current_supply)}\n"
            f"  Tokens added: Airdrop = {airdrop_tokens}, Expansion = {expansion_tokens}, Block rewards distributed = {int(block_reward_tokens)}"
        )
    
    # Next two years (include only expansion tokens + block rewards)
    for year in range(second_two_years):
        block_reward_tokens = min(yearly_block_reward, remaining_block_rewards)
        yearly_tokens_added = block_reward_tokens + expansion_tokens
        current_supply += yearly_tokens_added
        remaining_block_rewards -= block_reward_tokens  # Subtract what was distributed
        
        output.append(
            f"End of year {year + 3}:\n"
            f"  Total supply = {int(current_supply)}\n"
            f"  Tokens added: Expansion = {expansion_tokens}, Block rewards distributed = {int(block_reward_tokens)}"
        )
    
    # Year 5: Expansion tokens + inflation (1.75% of total supply at end of year 4 + expansion tokens)
    inflation_tokens_year_5 = inflation_rate * (current_supply + expansion_tokens)
    yearly_tokens_added = expansion_tokens + inflation_tokens_year_5
    current_supply += yearly_tokens_added
    
    output.append(
        f"End of year 5:\n"
        f"  Total supply = {int(current_supply)}\n"
        f"  Tokens added: Expansion = {expansion_tokens}, Inflation = {int(inflation_tokens_year_5)}"
    )
    
    # Year 6: Expansion tokens + inflation (1.75% of total supply at end of year 5 + expansion tokens)
    inflation_tokens_year_6 = inflation_rate * (current_supply + expansion_tokens)
    yearly_tokens_added = expansion_tokens + inflation_tokens_year_6
    current_supply += yearly_tokens_added
    
    output.append(
        f"End of year 6:\n"
        f"  Total supply = {int(current_supply)}\n"
        f"  Tokens added: Expansion = {expansion_tokens}, Inflation = {int(inflation_tokens_year_6)}"
    )

    # Year 7: Only inflation (1.75% of total supply at end of year 6)
    inflation_tokens_year_7 = inflation_rate * current_supply
    current_supply += inflation_tokens_year_7
    
    output.append(
        f"End of year 7:\n"
        f"  Total supply = {int(current_supply)}\n"
        f"  Tokens added: Inflation = {int(inflation_tokens_year_7)}"
    )
    
    return output

# Streamlit app layout
st.title("Token Supply Calculator")

# Date input
start_date_str = st.date_input("Select a starting date", value=datetime(2024, 11, 30))

# Button to calculate
if st.button("Calculate Token Supply"):
    try:
        result = calculate_token_supply(start_date_str)
        for line in result:
            st.write(line)
    except ValueError as e:
        st.error(str(e))
