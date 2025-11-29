import soccerdata as sd
import pandas as pd
import warnings

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- CONFIGURATION (Use the one you prefer) ---
# Start with Premier League only to verify it works fast.
leagues = ["Big 5 European Leagues Combined"] 
seasons = ['2023', '2024', '2025'] 

print(f"--- INITIALIZING SOCCERDATA (Leagues: {leagues}) ---")
fbref = sd.FBref(leagues=leagues, seasons=seasons)

# --- 1. GET BASIC SCHEDULE ---
print("1/4 Downloading Schedule...")
schedule = fbref.read_schedule().reset_index()

# Fix 'score' column (Split "2-1" into Home=2, Away=1)
if 'score' in schedule.columns:
    scores = schedule['score'].str.split(r'[-–]', expand=True)
    schedule['home_score'] = pd.to_numeric(scores[0], errors='coerce')
    schedule['away_score'] = pd.to_numeric(scores[1], errors='coerce')

# Select base columns (include 'game' for merging)
base_df = schedule[[
    'league', 'season', 'game', 'date', 'home_team', 'away_team',
    'home_score', 'away_score', 'home_xg', 'away_xg'
]]

# --- HELPER FUNCTION TO CLEAN STATS ---
def get_clean_stats(stat_type, valid_cols_map):
    """Downloads stats, flattens columns, and selects specific ones."""
    print(f"...Downloading {stat_type} stats...")
    df = fbref.read_team_match_stats(stat_type=stat_type).reset_index()
    
    # Flatten MultiIndex columns (e.g., ('Standard', 'Sh') -> 'Standard_Sh')
    new_cols = []
    for col in df.columns:
        if isinstance(col, tuple):
            # Join tuple parts, but ignore empty strings
            name = "_".join([str(c) for c in col if c]).strip()
            new_cols.append(name)
        else:
            new_cols.append(col)
    df.columns = new_cols

    # Rename specific columns we want based on the map
    # The map is { 'Current_Name': 'New_Name' }
    renamed_df = df.rename(columns=valid_cols_map)
    
    # Keep only the columns we successfully renamed + merge keys
    keep_cols = ['league', 'season', 'game', 'team'] + list(valid_cols_map.values())
    
    # Filter only columns that actually exist
    final_cols = [c for c in keep_cols if c in renamed_df.columns]
    return renamed_df[final_cols]

# --- 2. GET ADVANCED STATS ---
# Define the columns we want to extract
shooting_map = {
    'Standard_Sh': 'shots',
    'Standard_SoT': 'shots_on_target', 
    'Standard_Dist': 'shot_distance',
    'Standard_FK': 'free_kicks',
    'Expected_npxG': 'npxG'
}

defense_map = {
    'Tackles_TklW': 'tackles_won',
    'Pressures_Press': 'pressures',
    'Interceptions_Int': 'interceptions',
    'Aerial Duels_Won%': 'aerial_won_pct'
}

possession_map = {
    'Possession_Poss': 'possession_pct',
    'Touches_Att Pen': 'touches_in_opp_box',
    'Carries_PrgC': 'progressive_carries',
    'Passing_PrgP': 'progressive_passes'
}

print("2/4 Getting Stats (Shooting, Defense, Possession)...")
# Note: In new fbref, sometimes defense/possession columns change slightly.
# This script tries to grab the standard ones.
shooting = get_clean_stats("shooting", shooting_map)
defense = get_clean_stats("defense", defense_map)
possession = get_clean_stats("possession", possession_map)

# --- 3. MERGE STATS INTO ONE DATAFRAME ---
# We merge shooting + defense + possession into a single "Team Stats" dataframe first
team_stats = pd.merge(shooting, defense, on=['league', 'season', 'game', 'team'], how='outer')
team_stats = pd.merge(team_stats, possession, on=['league', 'season', 'game', 'team'], how='outer')

# --- 4. MERGE TEAM STATS ONTO SCHEDULE (Home & Away) ---
print("3/4 Merging everything together...")

# Merge Home Team Stats
master_df = pd.merge(
    base_df, 
    team_stats.add_prefix('home_'), 
    left_on=['league', 'season', 'game', 'home_team'], 
    right_on=['home_league', 'home_season', 'home_game', 'home_team'], 
    how='left'
)

# Merge Away Team Stats
master_df = pd.merge(
    master_df, 
    team_stats.add_prefix('away_'), 
    left_on=['league', 'season', 'game', 'away_team'], 
    right_on=['away_league', 'away_season', 'away_game', 'away_team'], 
    how='left'
)

# Clean up duplicate columns from merge
master_df = master_df.loc[:, ~master_df.columns.duplicated()]

# --- SAVE ---
file_name = "real_live_data_final.csv"
master_df.to_csv(file_name, index=False)
print(f"\nSUCCESS! Data saved to {file_name}")
print(f"Total Matches: {len(master_df)}")
print("Check the CSV to see your data.")