import soccerdata as sd
import pandas as pd
import warnings

# Suppress future warnings to keep output clean
warnings.simplefilter(action='ignore', category=FutureWarning)

leagues = ["Big 5 European Leagues Combined"] 
# 2026 is included as you correctly pointed out the data exists
seasons = ['2022', '2023', '2024', '2025', '2026']  

print(f"--- INITIALIZING SOCCERDATA (Leagues: {leagues}) ---")
# no_cache=True forces a fresh download to fix any broken/partial files from previous attempts
fbref = sd.FBref(leagues=leagues, seasons=seasons, no_cache=True)

print("1/2 Downloading Schedule & Scores (Fast)...")
schedule = fbref.read_schedule().reset_index()

print("2/2 Cleaning Data...")

# Split the score column if it exists
if 'score' in schedule.columns:
    scores = schedule['score'].str.split(r'[-–]', expand=True)
    schedule['home_score'] = pd.to_numeric(scores[0], errors='coerce')
    schedule['away_score'] = pd.to_numeric(scores[1], errors='coerce')

cols_to_keep = [
    'league', 
    'season', 
    'date', 
    'home_team', 
    'away_team',
    'home_score', 
    'away_score', 
    'home_xg', 
    'away_xg'
]

# Create a clean copy with only the columns we need
clean_df = schedule[cols_to_keep].copy()

# Remove rows where the match hasn't been played yet (no scores)
clean_df = clean_df.dropna(subset=['home_score', 'away_score'])

# Remove rows without xG data
clean_df = clean_df.dropna(subset=['home_xg', 'away_xg'])

# Convert date and sort
clean_df['date'] = pd.to_datetime(clean_df['date'])
clean_df = clean_df.sort_values('date')

print(f"📊 Rows after cleaning: {len(clean_df)}")

# --- 3. SAVE TO EXCEL ---
file_name = "real_live_data_filled.xlsx"
print(f"...Saving to {file_name}...")

clean_df.to_excel(file_name, index=False)

print("\n" + "="*40)
print("✅ SUCCESS!")
print(f"📁 Data saved to: {file_name}")
print(f"🔢 Total Matches: {len(clean_df)}")
print("="*40)
print("This file contains ONLY:")
print("- Date, Teams, League")
print("- Final Scores (Home/Away)")
print("- Expected Goals (xG)")
print("(Empty columns and half-time scores were removed to prevent errors)")