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
    summary_data["Daily block rewards"] = int(daily_reward)
    # Display remaining block rewards at the start date
    remaining_block_rewards_at_start = initial_block_rewards - tokens_added
    summary_data["Remaining block rewards at start date"] = int(remaining_block_rewards_at_start)
    
    # Display one quarter of the remaining block rewards
    one_quarter_remaining_rewards = remaining_block_rewards / 4
    summary_data["Remaining yearly block rewards"] = int(one_quarter_remaining_rewards)
    
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
    beginning_supply = initial_token_supply  # Start with initial supply for the first year
    
    # First two years (include airdrop tokens + expansion tokens + block rewards)
    for year in range(first_two_years):
        block_reward_tokens = min(yearly_block_reward, remaining_block_rewards)
        yearly_tokens_added = airdrop_tokens + block_reward_tokens + expansion_tokens
        end_of_year_supply = beginning_supply + yearly_tokens_added
        
        yearly_data.append({
            "Year": year + 1,
            "Beginning of Year Supply": int(beginning_supply),
            "End of Year Supply": int(end_of_year_supply),
            "Airdrop Tokens": airdrop_tokens,
            "Expansion Tokens": expansion_tokens,
            "Inflation Tokens": 0,  # No inflation tokens in first two years
            "Block Rewards": int(block_reward_tokens),
        })
        
        # Update supplies for next iteration
        beginning_supply = end_of_year_supply
        remaining_block_rewards -= block_reward_tokens  # Subtract what was distributed
    
    # Next two years (include only expansion tokens + block rewards)
    for year in range(second_two_years):
        block_reward_tokens = min(yearly_block_reward, remaining_block_rewards)
        yearly_tokens_added = block_reward_tokens + expansion_tokens
        end_of_year_supply = beginning_supply + yearly_tokens_added
        
        yearly_data.append({
            "Year": year + 3,
            "Beginning of Year Supply": int(beginning_supply),
            "End of Year Supply": int(end_of_year_supply),
            "Airdrop Tokens": 0,  # No airdrop in these years
            "Expansion Tokens": expansion_tokens,
            "Inflation Tokens": 0,  # No inflation tokens in these years
            "Block Rewards": int(block_reward_tokens),
        })
        
        # Update supplies for next iteration
        beginning_supply = end_of_year_supply
        remaining_block_rewards -= block_reward_tokens  # Subtract what was distributed
    
    # Year 5: Expansion tokens + inflation (1.75% of total supply at end of year 4 + expansion tokens)
    inflation_tokens_year_5 = inflation_rate * (beginning_supply + expansion_tokens)
    yearly_tokens_added = expansion_tokens + inflation_tokens_year_5
    end_of_year_supply = beginning_supply + yearly_tokens_added
    
    yearly_data.append({
        "Year": 5,
        "Beginning of Year Supply": int(beginning_supply),
        "End of Year Supply": int(end_of_year_supply),
        "Airdrop Tokens": 0,
        "Expansion Tokens": expansion_tokens,
        "Block Rewards": 0,  # No block rewards after year 4
        "Inflation Tokens": int(inflation_tokens_year_5)
    })
    
    # Year 6: Expansion tokens + inflation (1.75% of total supply at end of year 5 + expansion tokens)
    inflation_tokens_year_6 = inflation_rate * (end_of_year_supply + expansion_tokens)
    yearly_tokens_added = expansion_tokens + inflation_tokens_year_6
    beginning_supply = end_of_year_supply
    end_of_year_supply = beginning_supply + yearly_tokens_added
    
    yearly_data.append({
        "Year": 6,
        "Beginning of Year Supply": int(beginning_supply),
        "End of Year Supply": int(end_of_year_supply),
        "Airdrop Tokens": 0,
        "Expansion Tokens": expansion_tokens,
        "Block Rewards": 0,
        "Inflation Tokens": int(inflation_tokens_year_6)
    })

    # Year 7: Only inflation (1.75% of total supply at end of year 6)
    inflation_tokens_year_7 = inflation_rate * end_of_year_supply
    beginning_supply = end_of_year_supply
    end_of_year_supply = beginning_supply + inflation_tokens_year_7
    
    yearly_data.append({
        "Year": 7,
        "Beginning of Year Supply": int(beginning_supply),
        "End of Year Supply": int(end_of_year_supply),
        "Airdrop Tokens": 0,
        "Expansion Tokens": 0,  # No expansion tokens in year 7
        "Block Rewards": 0,
        "Inflation Tokens": int(inflation_tokens_year_7)
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
        
        # Reorder columns: switch Block Rewards and Expansion Tokens
        yearly_df = yearly_df[["Year", "Beginning of Year Supply", "End of Year Supply", 
                                "Airdrop Tokens", "Block Rewards", "Expansion Tokens", 
                                "Inflation Tokens"]]

        st.markdown("### Yearly Data")
        st.table(yearly_df)
        
    except ValueError as e:
        st.error(str(e))
