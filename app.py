import streamlit as st
import pandas as pd
import numpy as np
import joblib
import google.generativeai as genai
import re
import toml
import os
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Advanced Football Predictor Pro",
    page_icon="⚽",
    layout="wide"
)

# --- CSS STYLING ---
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #0e1117; font-weight: 800; margin-bottom: 1rem; }
    .sub-header { font-size: 1.5rem; font-weight: 600; color: #444; margin-top: 2rem; margin-bottom: 1rem; }
    .card-title { margin: 0; font-size: 1.2rem; opacity: 0.9; text-transform: uppercase; letter-spacing: 1px;}
    .card-value { margin: 10px 0; font-size: 2.8rem; font-weight: 800; }
    .stat-box { margin-bottom: 20px; }
    .stat-label { font-size: 14px; color: #555; margin-bottom: 2px; }
    .stat-value { font-size: 28px; font-weight: 800; color: #111; margin-top: 0px; }
    .match-log { font-family: 'Courier New', monospace; font-size: 0.9rem; white-space: pre-wrap; }
</style>
""", unsafe_allow_html=True)

# --- SECRETS & API ---
try:
    secrets = toml.load(".streamlit/secrets.toml")
    GEMINI_API_KEY = secrets['google']['gemini_api_key']
    genai.configure(api_key=GEMINI_API_KEY)
except:
    st.error("❌ Error: Secrets not found. Please create .streamlit/secrets.toml")
    st.stop()

# --- LOAD RESOURCES ---
@st.cache_resource
def load_model():
    try: return joblib.load("real_data_football_model.pkl")
    except: return None

@st.cache_data(ttl=3600)
def load_data():
    if os.path.exists("all_big5_leagues_data.xlsx"):
        df = pd.read_excel("all_big5_leagues_data.xlsx", engine='openpyxl')
    elif os.path.exists("all_big5_leagues_data.csv"):
        df = pd.read_csv("all_big5_leagues_data.csv")
    else: return pd.DataFrame()
    
    # Standardize Names
    df = df.rename(columns={
        'Date': 'date', 'Home': 'home_team', 'Away': 'away_team',
        'Home_Goals': 'home_score', 'Away_Goals': 'away_score',
        'Home_xG': 'home_xg', 'Away_xG': 'away_xg'
    })
    
    # Remove duplicates/junk
    cols = [c for c in df.columns if not re.search(r'\.\d+$', c)]
    df = df[cols]
    
    # Ensure numeric
    cols_to_num = ['home_score', 'away_score', 'home_xg', 'away_xg']
    for c in cols_to_num:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')

    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date')

model = load_model()
df = load_data()

# --- HELPER: FORMAT MATCH STRING ---
def get_match_string(matches, team_name):
    history_text = ""
    for _, m in matches.iterrows():
        date_str = m['date'].strftime('%Y-%m-%d')
        
        if m['home_team'] == team_name:
            venue = "(H)"
            opp = m['away_team']
            gf = m['home_score']; ga = m['away_score']
            xgf = m.get('home_xg', 0); xga = m.get('away_xg', 0)
        else:
            venue = "(A)"
            opp = m['home_team']
            gf = m['away_score']; ga = m['home_score']
            xgf = m.get('away_xg', 0); xga = m.get('home_xg', 0)
            
        if gf > ga: res = "W"
        elif gf < ga: res = "L"
        else: res = "D"
        
        history_text += f"- {date_str} {venue} vs {opp} ({res}): {int(gf)}-{int(ga)} (xG: {xgf:.2f} vs {xga:.2f})\n"
    return history_text

# --- GEMINI AI ---
def get_gemini_analysis(home_team, away_team, prediction_text, confidence, home_log, away_log, h2h_log, h2h_avg, h2h_pct):
    model_name = "models/gemini-2.5-flash" 
    
    prompt = f"""
    You are a Senior Football Betting Analyst. 
    
    **MATCH:** {home_team} vs {away_team}
    **MODEL PREDICTION:** {prediction_text} ({confidence}% Confidence)
    
    **REAL DATA (LAST 5 MATCHES):**
    **{home_team} Recent Form:**
    {home_log}
    **{away_team} Recent Form:**
    {away_log}
    **Head-to-Head (H2H):**
    {h2h_log}
    
    **INSTRUCTIONS:**
    1. **Deep Dive:** Look at the xG vs Actual Score. Is a team getting lucky?
    2. **Tactical Analysis:** Based on who they played recently, how will they match up?
    3. **Value Bets:** Given the data, what is the smartest bet?
    
    **FORMAT (Markdown):**
    ### 📝 The Analyst's Deep Dive
    [Detailed paragraph analysis]
    
    ### 🎯 Value Predictions
    *   **Best Bet:** [Prediction]
    *   **The "Smart" Pick:** [Value pick]
    *   **Reasoning:** [Why?]
    
    ### ⚠️ Risk & Tactical Analysis
    *   **Tactical Mismatch:** [Analysis]
    *   **H2H Factor:** [Analysis]

    ### 🏁 Final Verdict
    [One decisive sentence.]
    """
    try:
        model_ai = genai.GenerativeModel(model_name)
        response = model_ai.generate_content(prompt)
        return response.text
    except: return "⚠️ AI Analysis unavailable."

# --- PREDICTION ENGINE ---
def make_prediction(home_team, away_team):
    # 1. Get Data
    home_matches = df[((df['home_team'] == home_team) | (df['away_team'] == home_team))].tail(5)
    away_matches = df[((df['home_team'] == away_team) | (df['away_team'] == away_team))].tail(5)
    
    h2h_matches = df[
        ((df['home_team'] == home_team) & (df['away_team'] == away_team) | 
         (df['home_team'] == away_team) & (df['away_team'] == home_team))
    ].tail(10)
    
    if len(home_matches) == 0 or len(away_matches) == 0: 
        return None, None, f"Insufficient data for {home_team} or {away_team}", None, None, None

    # 2. Calculate Features
    feats = {}
    
    def get_form_stats(matches, team):
        gf, ga, xgf, xga, pts = [], [], [], [], []
        for _, m in matches.iterrows():
            if m['home_team'] == team:
                gf.append(m['home_score']); ga.append(m['away_score'])
                xgf.append(m['home_xg']); xga.append(m['away_xg'])
                pts.append(3 if m['home_score'] > m['away_score'] else 1 if m['home_score'] == m['away_score'] else 0)
            else:
                gf.append(m['away_score']); ga.append(m['home_score'])
                xgf.append(m['away_xg']); xga.append(m['home_xg'])
                pts.append(3 if m['away_score'] > m['home_score'] else 1 if m['away_score'] == m['home_score'] else 0)
        return np.mean(gf), np.mean(ga), np.mean(xgf), np.mean(xga), np.sum(pts)

    # Fill Form Features
    feats['home_form_goals_for_5'], feats['home_form_goals_against_5'], feats['home_form_xg_for_5'], feats['home_form_xg_against_5'], feats['home_form_points_5'] = get_form_stats(home_matches, home_team)
    feats['away_form_goals_for_5'], feats['away_form_goals_against_5'], feats['away_form_xg_for_5'], feats['away_form_xg_against_5'], feats['away_form_points_5'] = get_form_stats(away_matches, away_team)
    
    # Fill H2H Features
    if len(h2h_matches) > 0:
        h_wins = 0; a_wins = 0; total_goals = 0; h_team_goals = 0; a_team_goals = 0
        
        for _, m in h2h_matches.iterrows():
            total_goals += (m['home_score'] + m['away_score'])
            
            if m['home_team'] == home_team:
                h_team_goals += m['home_score']
                a_team_goals += m['away_score']
                if m['home_score'] > m['away_score']: h_wins += 1
                elif m['away_score'] > m['home_score']: a_wins += 1
            else:
                h_team_goals += m['away_score']
                a_team_goals += m['home_score']
                if m['away_score'] > m['home_score']: h_wins += 1
                elif m['home_score'] > m['away_score']: a_wins += 1
        
        feats['h2h_matches'] = len(h2h_matches)
        feats['h2h_home_win_pct'] = h_wins / len(h2h_matches)
        feats['h2h_away_win_pct'] = a_wins / len(h2h_matches)
        feats['h2h_avg_goals'] = total_goals / len(h2h_matches)
        feats['h2h_avg_home_goals'] = h_team_goals / len(h2h_matches)
        feats['h2h_avg_away_goals'] = a_team_goals / len(h2h_matches)
    else:
        feats['h2h_matches'] = 0; feats['h2h_home_win_pct'] = 0; feats['h2h_away_win_pct'] = 0
        feats['h2h_avg_goals'] = 0; feats['h2h_avg_home_goals'] = 0; feats['h2h_avg_away_goals'] = 0

    # DIRECT ASSIGNMENT (NO ROUNDING)
    # We use the raw float average we just calculated for the form
    feats['home_xg'] = feats['home_form_xg_for_5']
    feats['away_xg'] = feats['away_form_xg_for_5']
    
    # 3. Predict
    try:
        # Reorder columns to match training EXACTLY
        input_df = pd.DataFrame([feats])
        input_df = input_df[model.feature_names_in_]
        
        return {
            'pred': model.predict(input_df)[0],
            'prob': model.predict_proba(input_df)[0],
            'feats': feats,
            'h_match': home_matches,
            'a_match': away_matches,
            'h2h': h2h_matches
        }
    except Exception as e: 
        return None, None, f"Model Error: {str(e)}", None, None, None

# --- UI START ---
st.markdown("### ⚽ Advanced Football Predictor Pro")

if df.empty or model is None:
    st.error("System not ready. Check data and model.")
    st.stop()

# 1. Inputs
all_teams = sorted(pd.concat([df['home_team'], df['away_team']]).unique())
c1, c2 = st.columns(2)
with c1: home_team = st.selectbox("Home Team", all_teams, index=0)
with c2: away_team = st.selectbox("Away Team", all_teams, index=1)

# 2. Generate
if st.button("🚀 Generate Prediction", type="primary", use_container_width=True):
    if home_team == away_team: st.error("Select different teams.")
    else:
        with st.spinner("Analyzing Match Data..."):
            res = make_prediction(home_team, away_team)
            
            if isinstance(res, tuple) and res[0] is None: 
                st.error(f"Prediction Failed: {res[2]}")
            elif res is None:
                st.error("Unknown Error.")
            else:
                pred = res['pred']
                probs = res['prob']
                feats = res['feats']
                
                # Create Strings for AI
                h_log_str = get_match_string(res['h_match'], home_team)
                a_log_str = get_match_string(res['a_match'], away_team)
                h2h_log_str = get_match_string(res['h2h'], home_team)
                
                # --- CARD DISPLAY ---
                if pred == 1:
                    title = "HOME WIN PREDICTION"
                    verdict = f"{home_team} to Win"
                    bg_color = "#2e7d32" # Green
                    conf_score = probs[1] * 100
                    icon = "🏠"
                    ai_verdict_text = f"{home_team} Win"
                else:
                    title = "AWAY WIN / DRAW PREDICTION"
                    verdict = f"{away_team} +0.5 (Win/Draw)"
                    bg_color = "#4353cc" # Blue
                    conf_score = probs[0] * 100
                    icon = "🚌"
                    ai_verdict_text = f"{away_team} Win/Draw"

                st.markdown(f"""
                <div style="background-color: {bg_color}; color: white; padding: 30px; border-radius: 12px; margin-bottom: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 5px;">
                        <span style="font-size: 2rem;">{icon}</span>
                        <h3 class="card-title">{title}</h3>
                    </div>
                    <h1 class="card-value">{verdict}</h1>
                    <div style="margin-top: 15px;">
                        <div style="display:flex; justify-content:space-between; margin-bottom:5px; font-weight:600;">
                            <span>Confidence</span>
                            <span>{conf_score:.1f}%</span>
                        </div>
                        <div style="width:100%; background: rgba(255,255,255,0.3); height:10px; border-radius:5px; overflow:hidden;">
                            <div style="width:{conf_score}%; background: #ffffff; height:100%;"></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # --- TABS ---
                st.markdown('<div class="sub-header">📊 Match Analysis</div>', unsafe_allow_html=True)
                t1, t2, t3 = st.tabs(["Form Analysis", "H2H History", "🤖 AI Insights (Detailed)"])
                
                with t1:
                    c1, c2 = st.columns(2)
                    with c1: 
                        st.markdown(f"**{home_team} Last 5 Matches:**")
                        st.markdown(f"```\n{h_log_str}\n```")
                    with c2: 
                        st.markdown(f"**{away_team} Last 5 Matches:**")
                        st.markdown(f"```\n{a_log_str}\n```")

                with t2:
                    if len(res['h2h']) > 0:
                        st.write(f"**Last {len(res['h2h'])} Head-to-Head**")
                        st.markdown(f"```\n{h2h_log_str}\n```")
                        
                        met1, met2, met3 = st.columns(3)
                        met1.metric("H2H Home Wins", f"{feats['h2h_home_win_pct']*100:.0f}%")
                        met2.metric("H2H Away Wins", f"{feats['h2h_away_win_pct']*100:.0f}%")
                        met3.metric("Avg H2H Goals", f"{feats['h2h_avg_goals']:.2f}")
                    else:
                        st.info("No head-to-head history available.")

                with t3:
                    with st.spinner("Generating Analyst Report..."):
                        ai_out = get_gemini_analysis(
                            home_team, away_team, ai_verdict_text, int(conf_score),
                            h_log_str, a_log_str, h2h_log_str, 
                            f"{feats['h2h_avg_goals']:.2f}", f"{feats['h2h_home_win_pct']*100:.0f}%"
                        )
                        st.markdown(ai_out)

                # --- ADDITIONAL INSIGHTS ---
                st.markdown('<div class="sub-header">🔍 Additional Insights</div>', unsafe_allow_html=True)
                
                def stat_html(label, value):
                    return f"""<div class="stat-box"><p class="stat-label">{label}</p><p class="stat-value">{value}</p></div>"""
                
                ic1, ic2 = st.columns(2)
                with ic1:
                    st.markdown("**Key Statistics:**")
                    st.markdown(stat_html("Home Form (Last 5)", f"{int(feats['home_form_points_5'])} pts"), unsafe_allow_html=True)
                    st.markdown(stat_html("Away Form (Last 5)", f"{int(feats['away_form_points_5'])} pts"), unsafe_allow_html=True)
                    st.markdown(stat_html("Home xG Advantage", f"{(feats['home_xg'] - feats['away_xg']):.2f}"), unsafe_allow_html=True)
                
                with ic2:
                    st.markdown("**Performance Metrics:**")
                    st.markdown(stat_html("Home Goals/Game", f"{feats['home_form_goals_for_5']:.2f}"), unsafe_allow_html=True)
                    st.markdown(stat_html("Away Goals/Game", f"{feats['away_form_goals_for_5']:.2f}"), unsafe_allow_html=True)
                    st.markdown(stat_html("Defensive Ratio", f"{(feats['home_form_goals_against_5'] - feats['away_form_goals_against_5']):.2f}"), unsafe_allow_html=True)