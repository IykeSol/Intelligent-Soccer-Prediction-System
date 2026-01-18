from seleniumbase import SB
import pandas as pd
import io
import time
import random
import os

LEAGUES = {
    "Premier League": {"id": 9,  "slug": "Premier-League"},
    "La Liga":        {"id": 12, "slug": "La-Liga"},
    "Bundesliga":     {"id": 20, "slug": "Bundesliga"},
    "Serie A":        {"id": 11, "slug": "Serie-A"},
    "Ligue 1":        {"id": 13, "slug": "Ligue-1"}
}

SEASONS = [
    "2020-2021", "2021-2022", "2022-2023", 
    "2023-2024", "2024-2025", "2025-2026"
]

OUTPUT_FILE = "all_big5_leagues_data.xlsx"

print("--- STARTING ROBUST SCRAPER (SAVES PROGRESS) ---")
print("⚠️ This will take 10-15 minutes. Please be patient!")

# Load existing data if file exists so we don't start over
if os.path.exists(OUTPUT_FILE):
    try:
        existing_df = pd.read_excel(OUTPUT_FILE)
        print(f"🔄 Resuming... Found {len(existing_df)} matches already saved.")
    except:
        existing_df = pd.DataFrame()
else:
    existing_df = pd.DataFrame()

with SB(uc=True) as sb:
    for league_name, info in LEAGUES.items():
        for season in SEASONS:
            
            # Check if we already have this data to skip scraping it again
            if not existing_df.empty:
                check = existing_df[
                    (existing_df['League'] == league_name) & 
                    (existing_df['Season'] == season)
                ]
                if len(check) > 100:
                    print(f"⏩ Skipping {league_name} {season} (Already have {len(check)} rows)")
                    continue

            # Generate URL
            if season == "2025-2026":
                url = f"https://fbref.com/en/comps/{info['id']}/schedule/{info['slug']}-Scores-and-Fixtures"
            else:
                url = f"https://fbref.com/en/comps/{info['id']}/{season}/schedule/{season}-{info['slug']}-Scores-and-Fixtures"
            
            print(f"\n🌍 Scraping: {league_name} [{season}]...")
            
            try:
                sb.activate_cdp_mode(url)
                
                # Wait for table
                try:
                    sb.wait_for_selector("table.stats_table", timeout=20)
                    time.sleep(2)
                except:
                    print(f"   ❌ Timeout waiting for table.")
                    continue

                # Parse Table
                page_source = sb.get_page_source()
                dfs = pd.read_html(io.StringIO(page_source), match="Score")
                
                if len(dfs) > 0:
                    df = dfs[0]
                    
                    # Clean headers
                    if 'Score' in df.columns:
                        df = df[df['Score'] != 'Score']
                        df = df.dropna(subset=['Score'])
                    
                    # Columns
                    cols_to_keep = ['Date', 'Home', 'Score', 'Away', 'xG', 'xG.1', 'Attendance']
                    existing_cols = [c for c in cols_to_keep if c in df.columns]
                    clean_df = df[existing_cols].copy()
                    
                    # Add Info
                    clean_df.insert(0, 'League', league_name)
                    clean_df.insert(1, 'Season', season)
                    
                    # Rename
                    if 'xG.1' in clean_df.columns:
                        clean_df = clean_df.rename(columns={'xG': 'Home_xG', 'xG.1': 'Away_xG'})
                    
                    # Split Score
                    if 'Score' in clean_df.columns:
                        scores = clean_df['Score'].str.split(r'[-–]', expand=True)
                        if len(scores.columns) == 2:
                            clean_df['Home_Goals'] = pd.to_numeric(scores[0], errors='coerce')
                            clean_df['Away_Goals'] = pd.to_numeric(scores[1], errors='coerce')
                    
                    print(f"   ✅ Found {len(clean_df)} matches.")
                    
                    # --- INSTANT SAVE ---
                    if os.path.exists(OUTPUT_FILE):
                        current_file = pd.read_excel(OUTPUT_FILE)
                        updated_df = pd.concat([current_file, clean_df], ignore_index=True)
                    else:
                        updated_df = clean_df
                        
                    updated_df.to_excel(OUTPUT_FILE, index=False)
                    print(f"   💾 Saved! Total rows in file: {len(updated_df)}")
                    
                else:
                    print("   ❌ No data found on page.")
                
                time.sleep(random.uniform(3, 6))
                
            except Exception as e:
                print(f"   ❌ Error: {e}")

print("\n✅ SCRAPING COMPLETE.")