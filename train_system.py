import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import re

print("🎯 REAL DATA FOOTBALL MODEL TRAINING")
print("=" * 60)
print("📊 USING ONLY REAL DATA - NO ESTIMATES")
print("=" * 60)

# --- 1. LOAD AND FIX CSV STRUCTURE ---
print("...Loading and Fixing CSV Structure...")
df = pd.read_csv("real_live_data_filled.csv")

# Fix duplicate columns by keeping the LAST occurrence
columns_to_keep = []
seen_columns = set()

for col in reversed(df.columns):
    clean_col = re.sub(r'\.\d+$', '', col)
    if clean_col not in seen_columns:
        columns_to_keep.append(col)
        seen_columns.add(clean_col)

columns_to_keep.reverse()
df = df[columns_to_keep]

print(f"✅ Loaded {len(df)} rows")

# Clean data - only keep matches with real scores
df = df.dropna(subset=['home_score', 'away_score'])
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

print(f"📈 After cleaning: {len(df)} matches")
print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")

# --- 2. CREATE HEAD-TO-HEAD FEATURES ---
print("\n...Creating Head-to-Head Features...")

def calculate_h2h_features(df):
    """Calculate head-to-head history between teams using REAL data only"""
    h2h_features = []
    
    for idx, match in df.iterrows():
        home_team = match['home_team']
        away_team = match['away_team']
        match_date = match['date']
        
        # Get previous matches between these two teams
        previous_matches = df[
            ((df['home_team'] == home_team) & (df['away_team'] == away_team) |
             (df['home_team'] == away_team) & (df['away_team'] == home_team)) &
            (df['date'] < match_date)
        ].tail(10)  # Last 10 H2H matches
        
        # H2H stats - ONLY if we have real data
        if len(previous_matches) > 0:
            home_wins = 0
            away_wins = 0
            draws = 0
            home_goals = 0
            away_goals = 0
            
            for _, h2h_match in previous_matches.iterrows():
                if h2h_match['home_team'] == home_team:
                    home_goals += h2h_match['home_score']
                    away_goals += h2h_match['away_score']
                    if h2h_match['home_score'] > h2h_match['away_score']:
                        home_wins += 1
                    elif h2h_match['home_score'] < h2h_match['away_score']:
                        away_wins += 1
                    else:
                        draws += 1
                else:
                    home_goals += h2h_match['away_score']
                    away_goals += h2h_match['home_score']
                    if h2h_match['away_score'] > h2h_match['home_score']:
                        home_wins += 1
                    elif h2h_match['away_score'] < h2h_match['home_score']:
                        away_wins += 1
                    else:
                        draws += 1
            
            total_matches = len(previous_matches)
            h2h_features.append({
                'date': match_date,
                'home_team': home_team,
                'away_team': away_team,
                'h2h_matches': total_matches,
                'h2h_home_wins': home_wins,
                'h2h_away_wins': away_wins,
                'h2h_draws': draws,
                'h2h_home_win_pct': home_wins / total_matches,
                'h2h_away_win_pct': away_wins / total_matches,
                'h2h_avg_goals': (home_goals + away_goals) / total_matches,
                'h2h_avg_home_goals': home_goals / total_matches,
                'h2h_avg_away_goals': away_goals / total_matches,
            })
        else:
            # No previous H2H - use zeros (real absence of data)
            h2h_features.append({
                'date': match_date,
                'home_team': home_team,
                'away_team': away_team,
                'h2h_matches': 0,
                'h2h_home_wins': 0,
                'h2h_away_wins': 0,
                'h2h_draws': 0,
                'h2h_home_win_pct': 0,
                'h2h_away_win_pct': 0,
                'h2h_avg_goals': 0,
                'h2h_avg_home_goals': 0,
                'h2h_avg_away_goals': 0,
            })
    
    return pd.DataFrame(h2h_features)

h2h_df = calculate_h2h_features(df)
print(f"✅ Created H2H features for {len(h2h_df)} matches")

# --- 3. CREATE LAST 5 MATCHES FORM ---
print("\n...Creating Last 5 Matches Form...")

def calculate_team_form(df):
    """Calculate form from last 5 matches using REAL data only"""
    all_team_matches = []
    
    # Home matches - only use REAL data that exists
    home_matches = df[['date', 'home_team', 'home_score', 'away_score', 'home_xg', 'away_xg']].copy()
    home_matches.columns = ['date', 'team', 'goals_for', 'goals_against', 'xg_for', 'xg_against']
    home_matches['is_home'] = 1
    
    # Away matches - only use REAL data that exists
    away_matches = df[['date', 'away_team', 'away_score', 'home_score', 'away_xg', 'home_xg']].copy()
    away_matches.columns = ['date', 'team', 'goals_for', 'goals_against', 'xg_for', 'xg_against']
    away_matches['is_home'] = 0
    
    # Combine
    team_matches = pd.concat([home_matches, away_matches], ignore_index=True)
    team_matches = team_matches.sort_values(['team', 'date'])
    
    # Calculate rolling averages for last 5 matches - REAL DATA ONLY
    numeric_columns = ['goals_for', 'goals_against', 'xg_for', 'xg_against']
    
    for col in numeric_columns:
        team_matches[f'form_{col}_5'] = team_matches.groupby('team')[col].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
        )
    
    # Calculate points from REAL matches only
    team_matches['points'] = team_matches.apply(
        lambda x: 3 if x['goals_for'] > x['goals_against'] else 1 if x['goals_for'] == x['goals_against'] else 0, 
        axis=1
    )
    
    team_matches['form_points_5'] = team_matches.groupby('team')['points'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).sum()
    )
    
    team_matches = team_matches.drop('points', axis=1)
    
    return team_matches

team_form = calculate_team_form(df)
print(f"✅ Created form data for {len(team_form)} team matches")

# --- 4. MERGE ALL FEATURES ---
print("\n...Merging All Features...")

# Merge H2H features
master_df = pd.merge(df, h2h_df, on=['date', 'home_team', 'away_team'], how='inner')

# Merge home team form
home_form = team_form[['date', 'team', 'form_goals_for_5', 'form_goals_against_5', 
                      'form_xg_for_5', 'form_xg_against_5', 'form_points_5']].copy()
home_form.columns = ['date', 'home_team'] + [f'home_{col}' for col in home_form.columns if col not in ['date', 'team']]

master_df = pd.merge(master_df, home_form, on=['date', 'home_team'], how='inner')

# Merge away team form
away_form = team_form[['date', 'team', 'form_goals_for_5', 'form_goals_against_5', 
                      'form_xg_for_5', 'form_xg_against_5', 'form_points_5']].copy()
away_form.columns = ['date', 'away_team'] + [f'away_{col}' for col in away_form.columns if col not in ['date', 'team']]

master_df = pd.merge(master_df, away_form, on=['date', 'away_team'], how='inner')

print(f"📊 Final dataset: {len(master_df)} matches")

# --- 5. USE ONLY REAL DATA - NO FILLING ---
print("\n...Using Only Real Data (No Filling)...")

# We'll only use matches where we have ALL the features from real data
features = [
    # Home team recent form
    'home_form_goals_for_5', 'home_form_goals_against_5',
    'home_form_xg_for_5', 'home_form_xg_against_5', 'home_form_points_5',
    
    # Away team recent form
    'away_form_goals_for_5', 'away_form_goals_against_5',
    'away_form_xg_for_5', 'away_form_xg_against_5', 'away_form_points_5',
    
    # Head-to-head history
    'h2h_matches', 'h2h_home_win_pct', 'h2h_away_win_pct',
    'h2h_avg_goals', 'h2h_avg_home_goals', 'h2h_avg_away_goals',
    
    # Current match xG
    'home_xg', 'away_xg'
]

# Remove any matches that have NaN in any of our features
master_df = master_df.dropna(subset=features)

print(f"📊 After removing matches with missing features: {len(master_df)} matches")

# --- 6. DEFINE TARGET AND TRAIN ---
print("\n...Training Model...")
master_df['target'] = np.where(master_df['home_score'] > master_df['away_score'], 1, 0)

print(f"🎯 Target distribution: {master_df['target'].mean():.1%} home wins")

X = master_df[features]
y = master_df['target']

print(f"📊 Features shape: {X.shape}")
print(f"📊 Target shape: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"📊 Training set: {len(X_train)} matches")
print(f"📊 Test set: {len(X_test)} matches")

# Train model
rf_model = RandomForestClassifier(
    n_estimators=150,
    min_samples_split=10,
    max_depth=20,
    random_state=42
)

print("🤖 Training Random Forest model...")
rf_model.fit(X_train, y_train)

# --- 7. EVALUATE ---
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("✅ MODEL TRAINING COMPLETE!")
print("=" * 60)
print(f"📈 Accuracy: {accuracy:.2%}")
print(f"🔧 Features: {len(features)}")
print(f"📊 Total matches: {len(master_df)}")

# Feature importance
print(f"\n📊 FEATURE IMPORTANCE:")
importances = pd.Series(rf_model.feature_importances_, index=features)
for feature, importance in importances.sort_values(ascending=False).items():
    print(f"  {importance:.4f} - {feature}")

# --- 8. SAVE MODEL ---
joblib.dump(rf_model, "real_data_football_model.pkl")
print(f"\n💾 Model saved to 'real_data_football_model.pkl'")

# --- 9. TRAINING SUMMARY ---
print("\n" + "=" * 60)
print("📊 TRAINING SUMMARY - REAL DATA ONLY")
print("=" * 60)
print(f"🏆 Leagues: {len(master_df['league'].unique())}")
print(f"📅 Date range: {master_df['date'].min().strftime('%Y-%m-%d')} to {master_df['date'].max().strftime('%Y-%m-%d')}")
print(f"🎯 Home win rate: {master_df['target'].mean():.1%}")

print(f"\n🔑 REAL FEATURES USED:")
for feature in features:
    print(f"  ✅ {feature}")

print(f"\n🎉 READY FOR REAL PREDICTIONS!")
print(f"   Using ONLY real historical data - no estimates!")