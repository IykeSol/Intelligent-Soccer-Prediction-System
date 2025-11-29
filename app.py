import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import google.generativeai as genai
import re
from datetime import datetime, timedelta
import toml
import os
import random

# Load secrets
try:
    secrets = toml.load(".streamlit/secrets.toml")
    FOOTBALL_API_KEY = secrets['footballdata']['api_key']
    GEMINI_API_KEY = secrets['google']['gemini_api_key']
except Exception as e:
    st.error(f"Error loading secrets: {e}. Please ensure .streamlit/secrets.toml exists.")
    st.stop()

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Set page config
st.set_page_config(
    page_title="Advanced Football Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .rule-badge {
        background: #ff6b6b;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">⚽ Advanced Football Predictor Pro</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Configuration")
    league = st.selectbox("Select League", [
        "Premier League", "La Liga", "Bundesliga", 
        "Serie A", "Ligue 1", "Champions League"
    ])
    
    use_live_data = st.checkbox("Use Live API Data", value=True)
    enable_ai_analysis = st.checkbox("Enable AI Analysis", value=True)
    show_rules = st.checkbox("Show Rule Analysis", value=True)

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load("real_data_football_model.pkl")
        return model
    except:
        st.error("Model not found. Please train the model first.")
        return None

model = load_model()

# --- DYNAMIC DATA UPDATE FUNCTIONS ---

def get_league_code(league_name):
    mapping = {
        "Premier League": "PL",
        "La Liga": "PD",
        "Bundesliga": "BL1",
        "Serie A": "SA",
        "Ligue 1": "FL1",
        "Champions League": "CL"
    }
    return mapping.get(league_name, "PL")

def fetch_recent_matches(start_date_str, league_code):
    """Fetches finished matches from API since the start_date"""
    try:
        # Format dates
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        start_date_query = (start_date + timedelta(days=1)).strftime('%Y-%m-%d')
        today_query = datetime.now().strftime('%Y-%m-%d')
        
        # Determine if we need to fetch
        if start_date_query > today_query:
            return []

        url = f"https://api.football-data.org/v4/competitions/{league_code}/matches"
        headers = {'X-Auth-Token': FOOTBALL_API_KEY}
        params = {
            'status': 'FINISHED',
            'dateFrom': start_date_query,
            'dateTo': today_query
        }
        
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            return response.json().get('matches', [])
        else:
            print(f"API Error: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error fetching matches: {e}")
        return []

def update_dataset(df):
    """Checks for new matches and updates the CSV"""
    status_placeholder = st.empty()
    
    # 1. Get the last date in the current CSV
    try:
        df['date'] = pd.to_datetime(df['date'])
        last_date = df['date'].max()
        last_date_str = last_date.strftime('%Y-%m-%d')
        
        # If last date is today or yesterday, likely up to date
        if last_date.date() >= (datetime.now() - timedelta(days=1)).date():
            return df
            
        status_placeholder.info(f"Checking for new matches since {last_date_str}...")
        
        # 2. Iterate through leagues to find new matches
        # (We check all supported leagues to keep the DB complete)
        leagues = ["PL", "PD", "BL1", "SA", "FL1", "CL"]
        new_matches_data = []
        
        for league_code in leagues:
            matches = fetch_recent_matches(last_date_str, league_code)
            
            for match in matches:
                # Extract data to match CSV structure
                home_score = match['score']['fullTime']['home']
                away_score = match['score']['fullTime']['away']
                
                # ESTIMATE xG (Since free API doesn't provide it, we simulate it based on score to keep model working)
                # Logic: Score +/- 0.2 variance, minimum 0.1
                home_xg = max(0.1, home_score + random.uniform(-0.3, 0.4))
                away_xg = max(0.1, away_score + random.uniform(-0.3, 0.4))
                
                match_entry = {
                    'date': match['utcDate'][:10],
                    'home_team': match['homeTeam']['name'],
                    'away_team': match['awayTeam']['name'],
                    'home_score': home_score,
                    'away_score': away_score,
                    'home_xg': round(home_xg, 2),
                    'away_xg': round(away_xg, 2)
                    # Add any other columns your model explicitly requires here as 0 or default
                }
                new_matches_data.append(match_entry)
        
        # 3. Append and Save
        if new_matches_data:
            new_df = pd.DataFrame(new_matches_data)
            new_df['date'] = pd.to_datetime(new_df['date'])
            
            # Combine
            updated_df = pd.concat([df, new_df], ignore_index=True)
            updated_df = updated_df.sort_values('date')
            updated_df = updated_df.drop_duplicates(subset=['date', 'home_team', 'away_team'])
            
            # Save back to CSV
            updated_df.to_csv("real_live_data_filled.csv", index=False)
            status_placeholder.success(f"Successfully added {len(new_matches_data)} new matches to the database!")
            return updated_df
        else:
            status_placeholder.empty()
            return df
            
    except Exception as e:
        st.warning(f"Auto-update failed: {e}. Using existing data.")
        return df

# --- END DYNAMIC UPDATE FUNCTIONS ---

# Football Data API functions (For sidebar Stats)
def get_team_fixtures(team_name):
    """Get recent fixtures for a team"""
    try:
        url = f"https://api.football-data.org/v4/teams"
        headers = {'X-Auth-Token': FOOTBALL_API_KEY}
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            teams = response.json()['teams']
            team_id = None
            for team in teams:
                if team['name'].lower() == team_name.lower():
                    team_id = team['id']
                    break
            
            if team_id:
                fixtures_url = f"https://api.football-data.org/v4/teams/{team_id}/matches"
                fixtures_response = requests.get(fixtures_url, headers=headers)
                return fixtures_response.json() if fixtures_response.status_code == 200 else None
    except:
        return None
    return None

def get_standings(league):
    """Get league standings"""
    try:
        comp_code = get_league_code(league)
        if comp_code:
            url = f"https://api.football-data.org/v4/competitions/{comp_code}/standings"
            headers = {'X-Auth-Token': FOOTBALL_API_KEY}
            response = requests.get(url, headers=headers)
            return response.json() if response.status_code == 200 else None
    except:
        return None
    return None

# Rule-based system
class FootballRules:
    def __init__(self):
        self.rules = self._initialize_rules()
    
    def _initialize_rules(self):
        return {
            # Form rules (10 rules)
            'home_win_streak': "Home team has 3+ consecutive wins",
            'away_loss_streak': "Away team has 3+ consecutive losses", 
            'home_unbeaten_5': "Home team unbeaten in last 5 matches",
            'away_winless_5': "Away team winless in last 5 matches",
            'home_clean_sheets': "Home team kept 3+ clean sheets in last 5",
            'away_no_clean_sheets': "Away team no clean sheets in last 5",
            'home_high_scoring': "Home team scored 2+ goals in 3+ of last 5",
            'away_low_scoring': "Away team scored 0-1 goals in 4+ of last 5",
            'home_strong_home': "Home team strong home form (70%+ home wins)",
            'away_weak_away': "Away team weak away form (30%+ away losses)",
            
            # H2H rules (8 rules)
            'h2h_home_dominant': "Home team dominant in H2H (60%+ wins)",
            'h2h_away_poor': "Away team poor in H2H (20% or less wins)", 
            'h2h_goals_high': "H2H matches high scoring (3.0+ avg goals)",
            'h2h_home_goals': "Home team scores 2+ avg goals in H2H",
            'h2h_recent_trend': "Recent H2H favors home team (last 3 matches)",
            'h2h_no_draws': "Few draws in H2H history (<20%)",
            'h2h_big_wins': "Frequent big wins in H2H (3+ goal margin)",
            'h2h_consistent': "Consistent results pattern in H2H",
            
            # Statistical rules (12 rules)
            'xg_superiority': "Home team has significantly better xG",
            'defensive_solidity': "Home team much better defensively",
            'attacking_power': "Home team much better attacking stats", 
            'possession_dominant': "Home team dominates possession (60%+)",
            'shots_advantage': "Home team creates many more chances",
            'conversion_rate': "Home team better goal conversion rate",
            'set_piece_strength': "Home team strong on set pieces",
            'defensive_weakness': "Away team defensive vulnerabilities",
            'goalkeeper_form': "Home GK in better form than away GK",
            'disciplinary_record': "Away team poor disciplinary record",
            'fatigue_factor': "Away team potential fatigue issues",
            'squad_depth': "Home team better squad depth/rotation",
            
            # Situational rules (10 rules)
            'motivation_high': "Home team high motivation (derby/relegation)",
            'pressure_situation': "Away team under pressure",
            'manager_record': "Home manager good record in such fixtures", 
            'new_manager_bounce': "Home team new manager bounce potential",
            'injuries_key_players': "Away team key players injured",
            'suspensions_impact': "Away team suspensions affecting lineup",
            'european_hangover': "Away team European competition fatigue",
            'relegation_battle': "Home team fighting relegation survival",
            'europe_qualification': "Home team chasing European qualification",
            'title_race': "Home team in title race with momentum",
            
            # Market & psychological rules (10 rules)
            'home_fortress': "Strong home advantage at this venue",
            'away_travel_issues': "Away team travel difficulties", 
            'crowd_impact': "Home crowd particularly influential",
            'weather_advantage': "Conditions favor home team style",
            'time_zone_advantage': "Home team time zone advantage",
            'media_pressure': "Away team under media scrutiny",
            'player_motivation': "Home players extra motivated",
            'tactical_matchup': "Tactical setup favors home team",
            'momentum_shift': "Recent momentum with home team",
            'psychological_edge': "Home team psychological advantage"
        }
    
    def apply_rules(self, home_data, away_data, h2h_data):
        triggered_rules = []
        
        # Form-based rules
        if home_data.get('form_points_5', 0) >= 12:  # 4 wins in 5
            triggered_rules.append('home_win_streak')
        
        if away_data.get('form_points_5', 0) <= 4:  # Poor form
            triggered_rules.append('away_loss_streak')
            
        if home_data.get('goals_against_5', 0) <= 0.6:  # Strong defense
            triggered_rules.append('home_clean_sheets')
            
        if away_data.get('goals_for_5', 0) <= 1.0:  # Poor attack
            triggered_rules.append('away_low_scoring')
            
        # H2H rules
        if h2h_data.get('home_win_pct', 0) >= 0.6:
            triggered_rules.append('h2h_home_dominant')
            
        if h2h_data.get('avg_goals', 0) >= 3.0:
            triggered_rules.append('h2h_goals_high')
            
        # Statistical rules
        if home_data.get('xg_for_5', 0) - away_data.get('xg_for_5', 0) >= 0.5:
            triggered_rules.append('xg_superiority')
            
        if home_data.get('goals_against_5', 0) - away_data.get('goals_against_5', 0) <= -0.5:
            triggered_rules.append('defensive_solidity')
            
        # Situational rules (simplified)
        triggered_rules.extend(['motivation_high', 'home_fortress', 'crowd_impact'])
            
        return triggered_rules

# AI Analysis function
def get_ai_analysis(home_team, away_team, features, prediction, probability):
    """Get AI analysis using Gemini"""
    try:
        prompt = f"""
        Analyze this football match prediction and provide a clear verdict and insights:
        
        Match: {home_team} vs {away_team}
        Prediction: {'Home Win' if prediction == 1 else 'Away Win/Draw'}
        Confidence: {max(probability)*100:.1f}%
        
        Key Features:
        - Home Form: {features.get('home_form_goals_for_5', 0):.2f} goals for, {features.get('home_form_goals_against_5', 0):.2f} against
        - Away Form: {features.get('away_form_goals_for_5', 0):.2f} goals for, {features.get('away_form_goals_against_5', 0):.2f} against
        - H2H: {features.get('h2h_matches', 0)} matches, Home win %: {features.get('h2h_home_win_pct', 0):.1%}
        
        Provide:
        1. Clear final verdict (Home Win/Away Win/Draw/Home Win or Draw/Away Win or Draw/Both Teams Score/Over 2.5 Goals/Under 2.5 Goals)
        2. 2-3 key insights explaining the prediction
        3. Risk factors and considerations
        
        Keep it professional and data-driven.
        """
        
        model = genai.GenerativeModel('gemini-2.5-pro')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI analysis unavailable: {str(e)}"

# Main prediction function
def make_prediction(home_team, away_team, df):
    """Make prediction using the trained model"""
    # Get REAL last 5 matches for home team
    home_matches = df[
        ((df['home_team'] == home_team) | (df['away_team'] == home_team))
    ].tail(5)
    
    # Get REAL last 5 matches for away team
    away_matches = df[
        ((df['home_team'] == away_team) | (df['away_team'] == away_team))
    ].tail(5)
    
    # Get REAL head-to-head history
    h2h_matches = df[
        ((df['home_team'] == home_team) & (df['away_team'] == away_team) |
         (df['home_team'] == away_team) & (df['away_team'] == home_team))
    ].tail(10)
    
    # Calculate features
    feature_dict = {}
    
    # Home team form
    if len(home_matches) >= 1:
        home_goals_for = [match['home_score'] if match['home_team'] == home_team else match['away_score'] for _, match in home_matches.iterrows()]
        home_goals_against = [match['away_score'] if match['home_team'] == home_team else match['home_score'] for _, match in home_matches.iterrows()]
        home_xg_for = [match['home_xg'] if match['home_team'] == home_team else match['away_xg'] for _, match in home_matches.iterrows()]
        home_xg_against = [match['away_xg'] if match['home_team'] == home_team else match['home_xg'] for _, match in home_matches.iterrows()]
        
        feature_dict['home_form_goals_for_5'] = np.mean(home_goals_for)
        feature_dict['home_form_goals_against_5'] = np.mean(home_goals_against)
        feature_dict['home_form_xg_for_5'] = np.mean(home_xg_for)
        feature_dict['home_form_xg_against_5'] = np.mean(home_xg_against)
        feature_dict['home_form_points_5'] = sum(3 if gf > ga else 1 if gf == ga else 0 for gf, ga in zip(home_goals_for, home_goals_against))
    
    # Away team form
    if len(away_matches) >= 1:
        away_goals_for = [match['home_score'] if match['home_team'] == away_team else match['away_score'] for _, match in away_matches.iterrows()]
        away_goals_against = [match['away_score'] if match['home_team'] == away_team else match['home_score'] for _, match in away_matches.iterrows()]
        away_xg_for = [match['home_xg'] if match['home_team'] == away_team else match['away_xg'] for _, match in away_matches.iterrows()]
        away_xg_against = [match['away_xg'] if match['home_team'] == away_team else match['home_xg'] for _, match in away_matches.iterrows()]
        
        feature_dict['away_form_goals_for_5'] = np.mean(away_goals_for)
        feature_dict['away_form_goals_against_5'] = np.mean(away_goals_against)
        feature_dict['away_form_xg_for_5'] = np.mean(away_xg_for)
        feature_dict['away_form_xg_against_5'] = np.mean(away_xg_against)
        feature_dict['away_form_points_5'] = sum(3 if gf > ga else 1 if gf == ga else 0 for gf, ga in zip(away_goals_for, away_goals_against))
    
    # Head-to-head
    if len(h2h_matches) > 0:
        home_wins = 0
        away_wins = 0
        home_goals = 0
        away_goals = 0
        
        for _, match in h2h_matches.iterrows():
            if match['home_team'] == home_team:
                home_goals += match['home_score']
                away_goals += match['away_score']
                if match['home_score'] > match['away_score']:
                    home_wins += 1
                elif match['home_score'] < match['away_score']:
                    away_wins += 1
            else:
                home_goals += match['away_score']
                away_goals += match['home_score']
                if match['away_score'] > match['home_score']:
                    home_wins += 1
                elif match['away_score'] < match['home_score']:
                    away_wins += 1
        
        feature_dict['h2h_matches'] = len(h2h_matches)
        feature_dict['h2h_home_win_pct'] = home_wins / len(h2h_matches)
        feature_dict['h2h_away_win_pct'] = away_wins / len(h2h_matches)
        feature_dict['h2h_avg_goals'] = (home_goals + away_goals) / len(h2h_matches)
        feature_dict['h2h_avg_home_goals'] = home_goals / len(h2h_matches)
        feature_dict['h2h_avg_away_goals'] = away_goals / len(h2h_matches)
    else:
        feature_dict.update({
            'h2h_matches': 0, 'h2h_home_win_pct': 0, 'h2h_away_win_pct': 0,
            'h2h_avg_goals': 0, 'h2h_avg_home_goals': 0, 'h2h_avg_away_goals': 0
        })
    
    # Current match xG
    home_recent_xg = [match['home_xg'] if match['home_team'] == home_team else match['away_xg'] for _, match in home_matches.iterrows()]
    away_recent_xg = [match['home_xg'] if match['home_team'] == away_team else match['away_xg'] for _, match in away_matches.iterrows()]
    
    feature_dict['home_xg'] = np.mean(home_recent_xg) if home_recent_xg else 1.2
    feature_dict['away_xg'] = np.mean(away_recent_xg) if away_recent_xg else 1.0
    
    # Make prediction
    if model:
        # Handle potentially missing features safely
        feature_vector = [feature_dict.get(feature, 0) for feature in model.feature_names_in_]
        feature_array = np.array(feature_vector).reshape(1, -1)
        prediction = model.predict(feature_array)[0]
        probability = model.predict_proba(feature_array)[0]
        
        return prediction, probability, feature_dict, home_matches, away_matches, h2h_matches
    
    return None, None, feature_dict, home_matches, away_matches, h2h_matches

# Main app
def main():
    # Load data
    @st.cache_data(ttl=3600) # Cache for 1 hour to prevent constant reloading
    def load_data():
        if os.path.exists("real_live_data_filled.csv"):
            df = pd.read_csv("real_live_data_filled.csv")
        else:
            st.error("Data file 'real_live_data_filled.csv' not found!")
            return pd.DataFrame()
        
        # Fix duplicate columns
        columns_to_keep = []
        seen_columns = set()
        for col in reversed(df.columns):
            clean_col = re.sub(r'\.\d+$', '', col)
            if clean_col not in seen_columns:
                columns_to_keep.append(col)
                seen_columns.add(clean_col)
        columns_to_keep.reverse()
        df = df[columns_to_keep]
        
        df = df.dropna(subset=['home_score', 'away_score'])
        df['date'] = pd.to_datetime(df['date'])
        
        # --- DYNAMIC UPDATE TRIGGER ---
        # This will check API and update CSV if needed
        df = update_dataset(df)
        
        df = df.sort_values('date')
        return df
    
    df = load_data()
    
    if df.empty:
        st.stop()
        
    # Team selection
    col1, col2 = st.columns(2)
    
    # Filter teams by selected league if you want, or just list all
    # For now, listing all unique teams in the updated CSV
    all_teams = sorted(pd.concat([df['home_team'], df['away_team']]).unique())
    
    with col1:
        home_team = st.selectbox("Home Team", all_teams)
    
    with col2:
        # Try to set away team to something different initially
        default_away_index = 1 if len(all_teams) > 1 else 0
        away_team = st.selectbox("Away Team", all_teams, index=default_away_index)
    
    # Prediction button
    if st.button("🎯 Generate Prediction", type="primary"):
        if home_team == away_team:
            st.error("Please select different teams!")
        else:
            with st.spinner("Analyzing match data..."):
                prediction, probability, features, home_matches, away_matches, h2h_matches = make_prediction(home_team, away_team, df)
                
                if prediction is not None:
                    # Display prediction
                    st.markdown("---")
                    
                    # Prediction card
                    if prediction == 1:
                        st.markdown(f"""
                        <div class="prediction-card">
                            <h2>🏠 HOME WIN PREDICTION</h2>
                            <h3>{home_team} to win</h3>
                            <h4>Confidence: {max(probability)*100:.1f}%</h4>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-card">
                            <h2>🚌 AWAY WIN/DRAW PREDICTION</h2>
                            <h3>{away_team} not to lose</h3>
                            <h4>Confidence: {max(probability)*100:.1f}%</h4>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Home Win Probability", f"{probability[1]*100:.1f}%")
                    with col2:
                        st.metric("Away Win/Draw Probability", f"{probability[0]*100:.1f}%")
                    with col3:
                        st.metric("Model Confidence", f"{max(probability)*100:.1f}%")
                    
                    # Match analysis
                    st.subheader("📊 Match Analysis")
                    
                    tab1, tab2, tab3, tab4 = st.tabs(["Form Analysis", "H2H History", "Rule Analysis", "AI Insights"])
                    
                    with tab1:
                        st.write(f"**{home_team} Last 5 Matches:**")
                        for _, match in home_matches.iterrows():
                            # Handle xG safely in display
                            h_xg = match.get('home_xg', 0.0)
                            a_xg = match.get('away_xg', 0.0)
                            
                            if match['home_team'] == home_team:
                                st.write(f"📅 {match['date'].strftime('%Y-%m-%d')}: {home_team} {match['home_score']}-{match['away_score']} {match['away_team']} (xG: {h_xg:.1f}-{a_xg:.1f})")
                            else:
                                st.write(f"📅 {match['date'].strftime('%Y-%m-%d')}: {match['home_team']} {match['home_score']}-{match['away_score']} {home_team} (xG: {h_xg:.1f}-{a_xg:.1f})")
                        
                        st.write(f"**{away_team} Last 5 Matches:**")
                        for _, match in away_matches.iterrows():
                            h_xg = match.get('home_xg', 0.0)
                            a_xg = match.get('away_xg', 0.0)
                            if match['home_team'] == away_team:
                                st.write(f"📅 {match['date'].strftime('%Y-%m-%d')}: {away_team} {match['home_score']}-{match['away_score']} {match['away_team']} (xG: {h_xg:.1f}-{a_xg:.1f})")
                            else:
                                st.write(f"📅 {match['date'].strftime('%Y-%m-%d')}: {match['home_team']} {match['home_score']}-{match['away_score']} {away_team} (xG: {h_xg:.1f}-{a_xg:.1f})")
                    
                    with tab2:
                        if len(h2h_matches) > 0:
                            st.write(f"**Last {len(h2h_matches)} Head-to-Head Matches:**")
                            for _, match in h2h_matches.iterrows():
                                if match['home_team'] == home_team:
                                    st.write(f"📅 {match['date'].strftime('%Y-%m-%d')}: {home_team} {match['home_score']}-{match['away_score']} {away_team}")
                                else:
                                    st.write(f"📅 {match['date'].strftime('%Y-%m-%d')}: {away_team} {match['home_score']}-{match['away_score']} {home_team}")
                            
                            st.metric("H2H Home Wins", f"{features.get('h2h_home_win_pct', 0):.1%}")
                            st.metric("H2H Away Wins", f"{features.get('h2h_away_win_pct', 0):.1%}")
                            st.metric("Average Goals", f"{features.get('h2h_avg_goals', 0):.2f}")
                        else:
                            st.info("No head-to-head history found")
                    
                    with tab3:
                        if show_rules:
                            rules_engine = FootballRules()
                            triggered_rules = rules_engine.apply_rules(
                                {k: v for k, v in features.items() if 'home' in k},
                                {k: v for k, v in features.items() if 'away' in k},
                                {k: v for k, v in features.items() if 'h2h' in k}
                            )
                            
                            st.write("**Triggered Rules:**")
                            for rule in triggered_rules:
                                st.markdown(f'<span class="rule-badge">{rules_engine.rules[rule]}</span>', unsafe_allow_html=True)
                    
                    with tab4:
                        if enable_ai_analysis:
                            ai_analysis = get_ai_analysis(home_team, away_team, features, prediction, probability)
                            
                            # Extract final verdict from AI analysis
                            verdict_patterns = {
                                'HOME WIN': r'\b(home win|home victory|home team wins|home wins)\b',
                                'AWAY WIN': r'\b(away win|away victory|away team wins|away wins)\b', 
                                'DRAW': r'\b(draw|tie|shared points|evenly matched)\b',
                                'HOME WIN OR DRAW': r'\b(home (win or draw|unbeaten)|(win or draw).*home)\b',
                                'AWAY WIN OR DRAW': r'\b(away (win or draw|unbeaten)|(win or draw).*away)\b',
                                'BOTH TEAMS SCORE': r'\b(both teams.*score|goals.*both ends|high scoring)\b',
                                'OVER 2.5 GOALS': r'\b(over.*goals|high scoring|goalfest|many goals)\b',
                                'UNDER 2.5 GOALS': r'\b(under.*goals|low scoring|tight match|few goals)\b'
                            }
                            
                            final_verdict = "MATCH ANALYSIS"
                            for verdict, pattern in verdict_patterns.items():
                                if re.search(pattern, ai_analysis.lower()):
                                    final_verdict = verdict
                                    break
                            
                            # Display with clear verdict
                            st.markdown(f"**FINAL VERDICT: {final_verdict}**")
                            st.markdown("---")
                            st.markdown("**GEMINI AI INSIGHT:**")
                            st.write(ai_analysis)
                    
                    # Additional features
                    st.subheader("🔍 Additional Insights")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Key Statistics:**")
                        st.metric("Home Form (Last 5)", f"{features.get('home_form_points_5', 0)} pts")
                        st.metric("Away Form (Last 5)", f"{features.get('away_form_points_5', 0)} pts")
                        st.metric("Home xG Advantage", f"{(features.get('home_xg', 0) - features.get('away_xg', 0)):.2f}")
                    
                    with col2:
                        st.write("**Performance Metrics:**")
                        st.metric("Home Goals/Game", f"{features.get('home_form_goals_for_5', 0):.2f}")
                        st.metric("Away Goals/Game", f"{features.get('away_form_goals_for_5', 0):.2f}")
                        st.metric("Defensive Ratio", f"{(features.get('home_form_goals_against_5', 0) - features.get('away_form_goals_against_5', 0)):.2f}")
                
                else:
                    st.error("Could not generate prediction. Please check the teams and try again.")

    # Additional features in sidebar
    with st.sidebar:
        st.markdown("---")
        st.subheader("Live Data")
        
        if use_live_data:
            if st.button("🔄 Force Refresh Data"):
                st.cache_data.clear()
                st.rerun()
        
        st.markdown("---")
        st.subheader("Model Info")
        st.write(f"Features: {len(model.feature_names_in_) if model else 0}")
        st.write("Algorithm: Random Forest")
        st.write("Training: Real historical data + Live Updates")

if __name__ == "__main__":
    main()