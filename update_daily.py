from seleniumbase import SB
import pandas as pd
import io
import time
import os

# --- CONFIGURATION ---
LEAGUES = {
    "Premier League": "https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures",
    "La Liga":        "https://fbref.com/en/comps/12/schedule/La-Liga-Scores-and-Fixtures",
    "Bundesliga":     "https://fbref.com/en/comps/20/schedule/Bundesliga-Scores-and-Fixtures",
    "Serie A":        "https://fbref.com/en/comps/11/schedule/Serie-A-Scores-and-Fixtures",
    "Ligue 1":        "https://fbref.com/en/comps/13/schedule/Ligue-1-Scores-and-Fixtures"
}

FILE_NAME = "all_big5_leagues_data.xlsx"

print("--- ⚡ DAILY MATCH UPDATER (FBref Browser Mode) ⚡ ---")
print("This will verify all recent results and add missing games.")

new_matches = []

with SB(uc=True) as sb:
    for league_name, url in LEAGUES.items():
        print(f"\n🌍 Checking: {league_name}...")
        sb.activate_cdp_mode(url)
        
        try:
            # Wait for table
            sb.wait_for_selector("table.stats_table", timeout=15)
            time.sleep(2)
            
            # Get Data
            page_source = sb.get_page_source()
            dfs = pd.read_html(io.StringIO(page_source), match="Score")
            
            if len(dfs) > 0:
                df = dfs[0]
                
                # CLEANING
                if 'Score' in df.columns:
                    df = df[df['Score'] != 'Score'] # Remove headers
                    df = df.dropna(subset=['Score']) # Only keep COMPLETED games
                
                # Filter Columns
                cols_to_keep = ['Date', 'Home', 'Score', 'Away', 'xG', 'xG.1', 'Attendance']
                existing_cols = [c for c in cols_to_keep if c in df.columns]
                clean_df = df[existing_cols].copy()
                
                # Add Metadata
                clean_df.insert(0, 'League', league_name)
                clean_df.insert(1, 'Season', "2025-2026") # Current Season
                
                # Rename xG
                if 'xG.1' in clean_df.columns:
                    clean_df = clean_df.rename(columns={'xG': 'Home_xG', 'xG.1': 'Away_xG'})
                
                # Split Score
                if 'Score' in clean_df.columns:
                    scores = clean_df['Score'].str.split(r'[-–]', expand=True)
                    if len(scores.columns) == 2:
                        clean_df['Home_Goals'] = pd.to_numeric(scores[0], errors='coerce')
                        clean_df['Away_Goals'] = pd.to_numeric(scores[1], errors='coerce')
                
                # Convert Date for comparison
                clean_df['Date'] = pd.to_datetime(clean_df['Date'])
                
                print(f"   ✅ Retrieved {len(clean_df)} played matches.")
                new_matches.append(clean_df)
                
            else:
                print("   ❌ No table found.")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

# --- MERGE & SAVE ---
print("\n" + "="*40)
if len(new_matches) > 0:
    fetched_df = pd.concat(new_matches, ignore_index=True)
    
    if os.path.exists(FILE_NAME):
        # Load Existing
        existing_df = pd.read_excel(FILE_NAME)
        existing_df['Date'] = pd.to_datetime(existing_df['Date'])
        
        # Combine
        combined_df = pd.concat([existing_df, fetched_df], ignore_index=True)
        
        # REMOVE DUPLICATES (Critical Step)
        # We keep the 'last' entry, which updates any previously empty xG fields if they are now filled
        before_len = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=['Date', 'Home', 'Away'], keep='last')
        
        # Report Stats
        total_rows = len(combined_df)
        added_rows = total_rows - len(existing_df)
        
        # Save
        combined_df = combined_df.sort_values('Date')
        combined_df.to_excel(FILE_NAME, index=False)
        
        if added_rows > 0:
            print(f"✅ SUCCESS! Added {added_rows} new matches.")
        else:
            print(f"✅ Database verified. No new matches found (Already up to date).")
            
        print(f"📊 Total Matches in Database: {total_rows}")
        
    else:
        print("❌ Main database file not found. Running full save...")
        fetched_df.to_excel(FILE_NAME, index=False)
        print(f"✅ Created new database with {len(fetched_df)} matches.")
else:
    print("❌ Failed to fetch data.")
print("="*40)