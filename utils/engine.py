# utils/engine.py
import pandas as pd
import numpy as np
from scipy.stats import poisson
from datetime import timedelta

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
    if goal_diff <= 1: return 1.0
    elif goal_diff == 2: return 1.5
    else: return (11 + goal_diff) / 8.0

def expected_score(rating_a, rating_b, neutral=False):
    diff = (rating_b - rating_a) if neutral else (rating_b - rating_a) + HOME_ADVANTAGE
    return 1.0 / (1.0 + 10 ** (diff / ELO_SCALE))

def compute_elo_history(df, lookback_years=10):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    if lookback_years is not None:
        cutoff = df['date'].max() - timedelta(days=365 * lookback_years)
        df = df[df['date'] >= cutoff]
    
    teams = sorted(set(df['home_team']).union(set(df['away_team'])))
    ratings = {team: INITIAL_ELO for team in teams}
    history = []
    
    for _, row in df.iterrows():
        home, away = row['home_team'], row['away_team']
        home_goals, away_goals = row['home_score'], row['away_score']
        tournament = row['tournament']
        neutral = row.get('neutral', False)
        
        r_home, r_away = ratings[home], ratings[away]
        exp_home = expected_score(r_home, r_away, neutral=neutral)
        
        res_home = 1 if home_goals > away_goals else (0.5 if home_goals == away_goals else 0)
        
        weight = TOURNAMENT_WEIGHTS.get(tournament, 20)
        gd = abs(home_goals - away_goals)
        k = K_FACTOR_BASE * (weight / 30.0) * goal_diff_multiplier(gd)
        
        ratings[home] = r_home + k * (res_home - exp_home)
        ratings[away] = r_away + k * ((1 - res_home) - (1 - exp_home))
        
        history.append({
            'date': row['date'],
            'home_team': home,
            'away_team': away,
            'home_rating_after': ratings[home],
            'away_rating_after': ratings[away],
        })
    
    return ratings, pd.DataFrame(history)

def get_poisson_strengths(df, lookback_years):
    cutoff = df['date'].max() - timedelta(days=365 * lookback_years)
    df_filtered = df[df['date'] >= cutoff]
    
    home_matches = df_filtered.groupby('home_team').size()
    away_matches = df_filtered.groupby('away_team').size()
    matches = home_matches.add(away_matches, fill_value=0)
    
    home_goals = df_filtered.groupby('home_team')['home_score'].sum()
    away_goals = df_filtered.groupby('away_team')['away_score'].sum()
    goals_scored = home_goals.add(away_goals, fill_value=0)
    avg_scored = (goals_scored / matches).fillna(0)
    
    home_conceded = df_filtered.groupby('home_team')['away_score'].sum()
    away_conceded = df_filtered.groupby('away_team')['home_score'].sum()
    goals_conceded = home_conceded.add(away_conceded, fill_value=0)
    avg_conceded = (goals_conceded / matches).fillna(0)
    
    league_avg = (df_filtered['home_score'].sum() + df_filtered['away_score'].sum()) / (2 * len(df_filtered))
    
    attack = avg_scored / league_avg
    defense = avg_conceded / league_avg
    return attack, defense, league_avg

def predict_match(home, away, neutral, attack, defense, league_avg, elo_ratings):
    att_h, def_h = attack.get(home, 1.0), defense.get(home, 1.0)
    att_a, def_a = attack.get(away, 1.0), defense.get(away, 1.0)
    
    l_home = league_avg * att_h * def_a * (1.1 if not neutral else 1.0)
    l_away = league_avg * att_a * def_h * (0.9 if not neutral else 1.0)
    
    max_g = 10
    prob_matrix = np.outer(poisson.pmf(np.arange(max_g+1), l_home), poisson.pmf(np.arange(max_g+1), l_away))
    
    return {
        "l_home": l_home,
        "l_away": l_away,
        "p_home": np.sum(np.tril(prob_matrix, -1)),
        "p_draw": np.sum(np.diag(prob_matrix)),
        "p_away": np.sum(np.triu(prob_matrix, 1)),
        "matrix": prob_matrix,
        "elo_home": elo_ratings.get(home, INITIAL_ELO),
        "elo_away": elo_ratings.get(away, INITIAL_ELO)
    }

def monte_carlo(l_home, l_away, iterations=10000):
    h_goals = np.random.poisson(l_home, iterations)
    a_goals = np.random.poisson(l_away, iterations)
    return np.sum(h_goals > a_goals)/iterations, np.sum(h_goals == a_goals)/iterations, np.sum(h_goals < a_goals)/iterations
 