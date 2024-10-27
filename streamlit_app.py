import streamlit as st
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

# Constants
initial_token_supply = 2872024300
initial_block_rewards = 3175000000-11483890-2872024300
daily_average_reward=2000
daily_average_burn=300
airdrop_tokens = 95250000
expansion_tokens = 47625000
first_two_years = 2
second_two_years = 2
inflation_rate = 1.75 / 100  # 1.75% inflation
reference_date = datetime(2024, 10, 27, 0, 00)  # October 27th 0:00 UTC

def calculate_token_supply(start_date):
    # Convert start_date to datetime
    start_date = datetime.combine(start_date, datetime.min.time())
    
    # Calculate the number of full days between the reference date and the start date
    delta_days = (start_date - reference_date).days
    if delta_days < 0:
        raise ValueError("Start date must be after or on September 10th, 2024.")
    
    # Calculate tokens added due to daily reward
    tokens_added = delta_days * daily_average_reward
    tokens_removed=delta_days * daily_average_brun
    current_supply = initial_token_supply + tokens_added-tokens_removed
    
    # Calculate remaining block rewards
    remaining_block_rewards = initial_block_rewards - tokens_added
    
    # Divide block rewards evenly over 4 years
    yearly_block_reward = remaining_block_rewards / 4
    
    # Initialize output
    summary_data = {}
    
    # Initial information
    summary_data["Total supply on 09/10/2024, 0:00 UTC"] = int(initial_token_supply)
    summary_data["Remaining block rewards on 09/10/2024, 0:00 UTC"] = int(initial_block_rewards)
    summary_data["Daily block rewards"] = int(daily_average_reward)
    summary_data["Remaining block rewards at start date"] = int(initial_block_rewards - tokens_added)
    summary_data["Remaining yearly block rewards"] = int(yearly_block_reward)
    
    # Initialize yearly amounts
    summary_data["Yearly airdrop tokens"] = airdrop_tokens
    summary_data["Yearly expansion tokens"] = expansion_tokens
    
    # Initialize a DataFrame for yearly data
    yearly_data = []
    beginning_supply = current_supply  # Start with initial supply for the first year
    
    # Add a point for year 0
    yearly_data.append({
        "Year": 0,
        "Beginning of Year Supply": int(current_supply),  # This will be the starting point
        "End of Year Supply": int(current_supply),
        "Airdrop Tokens": 0,
        "Expansion Tokens": 0,
        "Inflation Tokens": 0,
        "Block Rewards": 0
    })

    # First two years (include airdrop tokens + expansion tokens + block rewards)
    for year in range(first_two_years):
        yearly_tokens_added = airdrop_tokens + yearly_block_reward + expansion_tokens
        end_of_year_supply = beginning_supply + yearly_tokens_added
        
        yearly_data.append({
            "Year": year + 1,
            "Beginning of Year Supply": int(beginning_supply),
            "End of Year Supply": int(end_of_year_supply),
            "Airdrop Tokens": airdrop_tokens,
            "Expansion Tokens": expansion_tokens,
            "Inflation Tokens": 0,
            "Block Rewards": int(yearly_block_reward),
        })
        
        beginning_supply = end_of_year_supply
        remaining_block_rewards -= yearly_block_reward
    
    # Next two years (include only expansion tokens + block rewards)
    for year in range(second_two_years):
        yearly_tokens_added = yearly_block_reward + expansion_tokens
        end_of_year_supply = beginning_supply + yearly_tokens_added
        
        yearly_data.append({
            "Year": year + 3,
            "Beginning of Year Supply": int(beginning_supply),
            "End of Year Supply": int(end_of_year_supply),
            "Airdrop Tokens": 0,
            "Expansion Tokens": expansion_tokens,
            "Inflation Tokens": 0,
            "Block Rewards": int(yearly_block_reward),
        })
        
        beginning_supply = end_of_year_supply
        remaining_block_rewards -= yearly_block_reward
    
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
        "Block Rewards": 0,
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
        "Expansion Tokens": 0,
        "Block Rewards": 0,
        "Inflation Tokens": int(inflation_tokens_year_7)
    })
    
    return summary_data, yearly_data

# Streamlit app layout
st.title("Token Supply Calculator")

# Display summary information before the date input
st.markdown("### Initial Information")
st.markdown(
    f"""
    <div style="border: 2px solid #1E90FF; padding: 10px; border-radius: 5px;">
    <ul>
        <li>Total supply on 09/10/2024, 0:00 UTC: {initial_token_supply}</li>
        <li>Remaining block rewards on 09/10/2024, 0:00 UTC: {initial_block_rewards}</li>
        <li>Daily average block rewards: {daily_averge_reward}</li>
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
        yearly_df = yearly_df[["Year", "Beginning of Year Supply", "End of Year Supply", 
                               "Airdrop Tokens", "Block Rewards", "Expansion Tokens", 
                               "Inflation Tokens"]]
        
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
