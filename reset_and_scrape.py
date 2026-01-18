from seleniumbase import SB
import pandas as pd
import io
import time
import random
import os

# --- CONFIGURATION ---
LEAGUES = {
    "Premier League": {"id": 9,  "slug": "Premier-League"},
    "La Liga":        {"id": 12, "slug": "La-Liga"},
    "Bundesliga":     {"id": 20, "slug": "Bundesliga"},
    "Serie A":        {"id": 11, "slug": "Serie-A"},
    "Ligue 1":        {"id": 13, "slug": "Ligue-1"}
}

# 6 Seasons x 5 Leagues = ~11,000 matches
SEASONS = [
    "2020-2021", "2021-2022", "2022-2023", 
    "2023-2024", "2024-2025", "2025-2026"
]

OUTPUT_FILE = "all_big5_leagues_data.xlsx"

print("--- 🔄 RESETTING DATA & STARTING SCRAPER ---")

# 1. FORCE DELETE THE OLD CORRUPT FILE
if os.path.exists(OUTPUT_FILE):
    try:
        os.remove(OUTPUT_FILE)
        print(f"🗑️ DELETED old corrupt file: {OUTPUT_FILE}")
    except PermissionError:
        print("❌ ERROR: Close Excel file before running this!")
        exit()

all_data = []

print("🚀 Starting Chrome... Please wait...")

with SB(uc=True) as sb:
    # 2. Open the first page to initialize the anti-bot bypass
    first_url = "https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures"
    sb.activate_cdp_mode(first_url)
    time.sleep(4)

    for league_name, info in LEAGUES.items():
        for season in SEASONS:
            
            # Construct URL
            if season == "2025-2026":
                url = f"https://fbref.com/en/comps/{info['id']}/schedule/{info['slug']}-Scores-and-Fixtures"
            else:
                url = f"https://fbref.com/en/comps/{info['id']}/{season}/schedule/{season}-{info['slug']}-Scores-and-Fixtures"
            
            print(f"\n🌍 Navigating to: {league_name} [{season}]")
            
            # 3. USE sb.open() TO ENSURE NAVIGATION HAPPENS
            sb.open(url)
            
            # 4. WAIT FOR THE SPECIFIC TABLE TO LOAD
            try:
                sb.wait_for_selector("table.stats_table", timeout=10)
                time.sleep(2) # Let the content settle
            except:
                print(f"   ❌ Failed to load table for {league_name} {season}")
                continue

            # 5. VERIFY WE ARE ON THE RIGHT PAGE
            # (Optional check to ensure we didn't get stuck)
            current_url = sb.get_current_url()
            if str(info['id']) not in current_url:
                print(f"   ⚠️ Warning: URL looks wrong. Expected ID {info['id']}, got {current_url}")

            # 6. EXTRACT DATA
            try:
                page_source = sb.get_page_source()
                dfs = pd.read_html(io.StringIO(page_source), match="Score")
                
                if len(dfs) > 0:
                    df = dfs[0]
                    
                    # Clean headers
                    if 'Score' in df.columns:
                        df = df[df['Score'] != 'Score']
                        df = df.dropna(subset=['Score'])
                    
                    cols_to_keep = ['Date', 'Home', 'Score', 'Away', 'xG', 'xG.1', 'Attendance']
                    existing_cols = [c for c in cols_to_keep if c in df.columns]
                    clean_df = df[existing_cols].copy()
                    
                    # Add Info
                    clean_df.insert(0, 'League', league_name)
                    clean_df.insert(1, 'Season', season)
                    
                    # Rename xG
                    if 'xG.1' in clean_df.columns:
                        clean_df = clean_df.rename(columns={'xG': 'Home_xG', 'xG.1': 'Away_xG'})
                    
                    # Split Score
                    if 'Score' in clean_df.columns:
                        scores = clean_df['Score'].str.split(r'[-–]', expand=True)
                        if len(scores.columns) == 2:
                            clean_df['Home_Goals'] = pd.to_numeric(scores[0], errors='coerce')
                            clean_df['Away_Goals'] = pd.to_numeric(scores[1], errors='coerce')
                    
                    print(f"   ✅ Collected {len(clean_df)} matches.")
                    all_data.append(clean_df)
                    
                else:
                    print("   ❌ Table found but Pandas couldn't read it.")
                
                time.sleep(random.uniform(3, 5))
                
            except Exception as e:
                print(f"   ❌ Error: {e}")

    # --- SAVE AT THE END ---
    print("\n" + "="*40)
    if len(all_data) > 0:
        final_df = pd.concat(all_data, ignore_index=True)
        
        if 'Date' in final_df.columns:
            final_df['Date'] = pd.to_datetime(final_df['Date'], errors='coerce')
            final_df = final_df.sort_values('Date')
        
        final_df.to_excel(OUTPUT_FILE, index=False)
        print("✅ SUCCESS! FILE SAVED.")
        print(f"📁 File: {OUTPUT_FILE}")
        print(f"🔢 Total Matches: {len(final_df)}")
        print("   (This number should be around 10,000 - 11,000)")
    else:
        print("❌ Failed. No data collected.")
    print("="*40)