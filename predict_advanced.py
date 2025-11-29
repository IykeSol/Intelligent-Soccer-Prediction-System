import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import re

def predict_real_data():
    print("🎯 REAL DATA FOOTBALL PREDICTOR")
    print("=" * 60)
    print("📊 USING ONLY REAL HISTORICAL DATA")
    print("=" * 60)
    
    # Load model
    try:
        model = joblib.load("real_data_football_model.pkl")
        trained_features = model.feature_names_in_
        print(f"✅ Model loaded with {len(trained_features)} features")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Please run train_system.py first to train the model")
        return
    
    # Load data
    df = pd.read_csv("real_live_data_filled.csv")
    
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
    df = df.sort_values('date')
    
    print(f"✅ Data loaded: {len(df)} matches")
    
    # User input
    print(f"\n🏟️  ENTER MATCH DETAILS")
    print("-" * 40)
    
    home_team = input("Enter HOME team: ").strip()
    away_team = input("Enter AWAY team: ").strip()
    
    print(f"\n📊 ANALYZING REAL DATA FOR {home_team} vs {away_team}")
    print("-" * 50)
    
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
    
    # Calculate features from REAL data only
    feature_dict = {}
    
    # Home team form - REAL DATA
    if len(home_matches) >= 1:  # Need at least 1 match for form
        home_goals_for = []
        home_goals_against = []
        home_xg_for = []
        home_xg_against = []
        home_points = []
        
        for _, match in home_matches.iterrows():
            if match['home_team'] == home_team:
                home_goals_for.append(match['home_score'])
                home_goals_against.append(match['away_score'])
                home_xg_for.append(match['home_xg'])
                home_xg_against.append(match['away_xg'])
                if match['home_score'] > match['away_score']:
                    home_points.append(3)
                elif match['home_score'] == match['away_score']:
                    home_points.append(1)
                else:
                    home_points.append(0)
            else:
                home_goals_for.append(match['away_score'])
                home_goals_against.append(match['home_score'])
                home_xg_for.append(match['away_xg'])
                home_xg_against.append(match['home_xg'])
                if match['away_score'] > match['home_score']:
                    home_points.append(3)
                elif match['away_score'] == match['home_score']:
                    home_points.append(1)
                else:
                    home_points.append(0)
        
        feature_dict['home_form_goals_for_5'] = np.mean(home_goals_for)
        feature_dict['home_form_goals_against_5'] = np.mean(home_goals_against)
        feature_dict['home_form_xg_for_5'] = np.mean(home_xg_for)
        feature_dict['home_form_xg_against_5'] = np.mean(home_xg_against)
        feature_dict['home_form_points_5'] = np.sum(home_points)
    else:
        print(f"❌ Not enough real data for {home_team}")
        return
    
    # Away team form - REAL DATA
    if len(away_matches) >= 1:  # Need at least 1 match for form
        away_goals_for = []
        away_goals_against = []
        away_xg_for = []
        away_xg_against = []
        away_points = []
        
        for _, match in away_matches.iterrows():
            if match['home_team'] == away_team:
                away_goals_for.append(match['home_score'])
                away_goals_against.append(match['away_score'])
                away_xg_for.append(match['home_xg'])
                away_xg_against.append(match['away_xg'])
                if match['home_score'] > match['away_score']:
                    away_points.append(3)
                elif match['home_score'] == match['away_score']:
                    away_points.append(1)
                else:
                    away_points.append(0)
            else:
                away_goals_for.append(match['away_score'])
                away_goals_against.append(match['home_score'])
                away_xg_for.append(match['away_xg'])
                away_xg_against.append(match['home_xg'])
                if match['away_score'] > match['home_score']:
                    away_points.append(3)
                elif match['away_score'] == match['home_score']:
                    away_points.append(1)
                else:
                    away_points.append(0)
        
        feature_dict['away_form_goals_for_5'] = np.mean(away_goals_for)
        feature_dict['away_form_goals_against_5'] = np.mean(away_goals_against)
        feature_dict['away_form_xg_for_5'] = np.mean(away_xg_for)
        feature_dict['away_form_xg_against_5'] = np.mean(away_xg_against)
        feature_dict['away_form_points_5'] = np.sum(away_points)
    else:
        print(f"❌ Not enough real data for {away_team}")
        return
    
    # Head-to-head - REAL DATA
    h2h_home_wins = 0
    h2h_away_wins = 0
    h2h_draws = 0
    h2h_home_goals = 0
    h2h_away_goals = 0
    
    if len(h2h_matches) > 0:
        for _, match in h2h_matches.iterrows():
            if match['home_team'] == home_team:
                h2h_home_goals += match['home_score']
                h2h_away_goals += match['away_score']
                if match['home_score'] > match['away_score']:
                    h2h_home_wins += 1
                elif match['home_score'] < match['away_score']:
                    h2h_away_wins += 1
                else:
                    h2h_draws += 1
            else:
                h2h_home_goals += match['away_score']
                h2h_away_goals += match['home_score']
                if match['away_score'] > match['home_score']:
                    h2h_home_wins += 1
                elif match['away_score'] < match['home_score']:
                    h2h_away_wins += 1
                else:
                    h2h_draws += 1
        
        feature_dict['h2h_matches'] = len(h2h_matches)
        feature_dict['h2h_home_win_pct'] = h2h_home_wins / len(h2h_matches)
        feature_dict['h2h_away_win_pct'] = h2h_away_wins / len(h2h_matches)
        feature_dict['h2h_avg_goals'] = (h2h_home_goals + h2h_away_goals) / len(h2h_matches)
        feature_dict['h2h_avg_home_goals'] = h2h_home_goals / len(h2h_matches)
        feature_dict['h2h_avg_away_goals'] = h2h_away_goals / len(h2h_matches)
    else:
        # No H2H history - use zeros (real absence of data)
        feature_dict.update({
            'h2h_matches': 0, 'h2h_home_win_pct': 0, 'h2h_away_win_pct': 0,
            'h2h_avg_goals': 0, 'h2h_avg_home_goals': 0, 'h2h_avg_away_goals': 0
        })
    
    # Current match xG - use recent averages from REAL data
    # Get average xG from last 5 matches for both teams
    home_recent_xg = [match['home_xg'] if match['home_team'] == home_team else match['away_xg'] 
                     for _, match in home_matches.iterrows()]
    away_recent_xg = [match['home_xg'] if match['home_team'] == away_team else match['away_xg'] 
                     for _, match in away_matches.iterrows()]
    
    feature_dict['home_xg'] = np.mean(home_recent_xg) if home_recent_xg else 1.2
    feature_dict['away_xg'] = np.mean(away_recent_xg) if away_recent_xg else 1.0
    
    # Display REAL data analysis
    print(f"\n📊 {home_team} LAST {len(home_matches)} REAL MATCHES:")
    for i, match in home_matches.iterrows():
        if match['home_team'] == home_team:
            print(f"   {match['date'].strftime('%Y-%m-%d')}: {home_team} {match['home_score']}-{match['away_score']} {match['away_team']} (xG: {match['home_xg']:.1f}-{match['away_xg']:.1f})")
        else:
            print(f"   {match['date'].strftime('%Y-%m-%d')}: {match['home_team']} {match['home_score']}-{match['away_score']} {home_team} (xG: {match['home_xg']:.1f}-{match['away_xg']:.1f})")
    
    print(f"\n   📈 FORM SUMMARY:")
    print(f"      Goals: {feature_dict['home_form_goals_for_5']:.2f} for, {feature_dict['home_form_goals_against_5']:.2f} against")
    print(f"      xG: {feature_dict['home_form_xg_for_5']:.2f} for, {feature_dict['home_form_xg_against_5']:.2f} against")
    print(f"      Points: {feature_dict['home_form_points_5']}")
    
    print(f"\n📊 {away_team} LAST {len(away_matches)} REAL MATCHES:")
    for i, match in away_matches.iterrows():
        if match['home_team'] == away_team:
            print(f"   {match['date'].strftime('%Y-%m-%d')}: {away_team} {match['home_score']}-{match['away_score']} {match['away_team']} (xG: {match['home_xg']:.1f}-{match['away_xg']:.1f})")
        else:
            print(f"   {match['date'].strftime('%Y-%m-%d')}: {match['home_team']} {match['home_score']}-{match['away_score']} {away_team} (xG: {match['home_xg']:.1f}-{match['away_xg']:.1f})")
    
    print(f"\n   📈 FORM SUMMARY:")
    print(f"      Goals: {feature_dict['away_form_goals_for_5']:.2f} for, {feature_dict['away_form_goals_against_5']:.2f} against")
    print(f"      xG: {feature_dict['away_form_xg_for_5']:.2f} for, {feature_dict['away_form_xg_against_5']:.2f} against")
    print(f"      Points: {feature_dict['away_form_points_5']}")
    
    if len(h2h_matches) > 0:
        print(f"\n🤝 REAL HEAD-TO-HEAD HISTORY ({len(h2h_matches)} matches):")
        for i, match in h2h_matches.iterrows():
            if match['home_team'] == home_team:
                print(f"   {match['date'].strftime('%Y-%m-%d')}: {home_team} {match['home_score']}-{match['away_score']} {away_team}")
            else:
                print(f"   {match['date'].strftime('%Y-%m-%d')}: {away_team} {match['home_score']}-{match['away_score']} {home_team}")
        
        print(f"\n   📊 H2H SUMMARY:")
        print(f"      {home_team} wins: {feature_dict['h2h_home_win_pct']:.1%} ({h2h_home_wins} matches)")
        print(f"      {away_team} wins: {feature_dict['h2h_away_win_pct']:.1%} ({h2h_away_wins} matches)")
        print(f"      Draws: {h2h_draws} matches")
        print(f"      Avg goals: {feature_dict['h2h_avg_goals']:.2f} per match")
    else:
        print(f"\n🤝 HEAD-TO-HEAD: No previous matches found")
    
    # Build feature vector for prediction
    feature_vector = []
    for feature in trained_features:
        if feature in feature_dict:
            feature_vector.append(feature_dict[feature])
        else:
            print(f"❌ Missing feature: {feature}")
            return
    
    # Make prediction
    feature_array = np.array(feature_vector).reshape(1, -1)
    prediction = model.predict(feature_array)[0]
    probability = model.predict_proba(feature_array)[0]
    
    # Display prediction
    print(f"\n" + "=" * 50)
    print("🎯 REAL DATA PREDICTION")
    print("=" * 50)
    
    if prediction == 1:
        print(f"   🏠 HOME WIN - {home_team} to win")
        confidence = probability[1] * 100
    else:
        print(f"   🚌 AWAY WIN or DRAW - {away_team} not to lose")
        confidence = probability[0] * 100
    
    print(f"\n📈 PROBABILITIES:")
    print(f"   🏠 {home_team} Win: {probability[1]*100:.1f}%")
    print(f"   🚌 {away_team} Win/Draw: {probability[0]*100:.1f}%")
    print(f"   🔥 Confidence: {max(probability)*100:.1f}%")
    
    print(f"\n💡 BETTING RECOMMENDATION:")
    if confidence > 75:
        print("   ✅ STRONG BET - Very high confidence")
    elif confidence > 65:
        print("   ✅ GOOD BET - High confidence") 
    elif confidence > 55:
        print("   ⚠️  MODERATE BET - Medium confidence")
    else:
        print("   🎲 NO BET - Low confidence")
    
    print(f"\n🔧 PREDICTION DETAILS:")
    print(f"   • 📊 Based on {len(home_matches) + len(away_matches)} real matches")
    print(f"   • 🤝 {len(h2h_matches)} head-to-head matches")
    print(f"   • 📈 Goals, xG, and points data")

if __name__ == "__main__":
    predict_real_data()