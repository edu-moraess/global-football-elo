# elo_engine.py - Versão final com janela deslizante e compatibilidade total
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

INITIAL_ELO = 1500
K_FACTOR_BASE = 32
ELO_SCALE = 400
HOME_ADVANTAGE = 50

TOURNAMENT_WEIGHTS = {
    'FIFA World Cup': 60,
    'Copa América': 50,
    'UEFA Euro': 50,
    'FIFA World Cup qualification': 40,
    'UEFA Euro qualification': 35,
    'CONMEBOL qualification': 35,
    'AFC Asian Cup': 30,
    'Africa Cup of Nations': 30,
    'CONCACAF Gold Cup': 30,
    'OFC Nations Cup': 25,
    'Friendly': 20,
}

def goal_diff_multiplier(goal_diff):
    if goal_diff == 1:
        return 1.0
    elif goal_diff == 2:
        return 1.5
    else:
        return (11 + goal_diff) / 8.0

def expected_score(rating_a, rating_b, neutral=False):
    if neutral:
        diff = rating_b - rating_a
    else:
        diff = (rating_b - rating_a) + HOME_ADVANTAGE
    return 1.0 / (1.0 + 10 ** (diff / ELO_SCALE))

def compute_elo_history(df, lookback_years=10):
    """
    Retorna (ratings_dict, history_df)
    Se lookback_years = None, usa todo histórico.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    if lookback_years is not None:
        cutoff = df['date'].max() - timedelta(days=365*lookback_years)
        df = df[df['date'] >= cutoff]
        print(f"[Elo] Usando dados desde {cutoff.date()} (últimos {lookback_years} anos)")
    
    teams = set(df['home_team']).union(set(df['away_team']))
    ratings = {team: INITIAL_ELO for team in teams}
    history = []
    
    for _, row in df.iterrows():
        home, away = row['home_team'], row['away_team']
        home_goals, away_goals = row['home_score'], row['away_score']
        tournament = row['tournament']
        neutral = row.get('neutral', False)
        
        r_home, r_away = ratings[home], ratings[away]
        exp_home = expected_score(r_home, r_away, neutral=neutral)
        exp_away = 1 - exp_home
        
        if home_goals > away_goals:
            res_home, res_away = 1, 0
        elif home_goals == away_goals:
            res_home, res_away = 0.5, 0.5
        else:
            res_home, res_away = 0, 1
        
        weight = TOURNAMENT_WEIGHTS.get(tournament, 20)
        gd = abs(home_goals - away_goals)
        k = K_FACTOR_BASE * (weight / 30.0) * goal_diff_multiplier(gd)
        
        ratings[home] = r_home + k * (res_home - exp_home)
        ratings[away] = r_away + k * (res_away - exp_away)
        
        history.append({
            'date': row['date'],
            'home_team': home,
            'away_team': away,
            'home_rating_before': r_home,
            'away_rating_before': r_away,
            'home_rating_after': ratings[home],
            'away_rating_after': ratings[away],
        })
    
    return ratings, pd.DataFrame(history)