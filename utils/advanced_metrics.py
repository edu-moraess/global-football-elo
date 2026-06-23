# utils/advanced_metrics.py — Métricas Quant Avançadas (Fora dos Clichês)
import pandas as pd
import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter1d

def get_team_history(history_df, team, n_last=50):
    """Extrai série temporal de ratings para um time."""
    mask = (history_df['home_team'] == team) | (history_df['away_team'] == team)
    h = history_df[mask].copy()
    if h.empty:
        return pd.DataFrame()
    h = h.sort_values('date')
    series = []
    for _, row in h.iterrows():
        if row['home_team'] == team:
            series.append({'date': row['date'], 'rating': row['home_rating_after'],
                           'expected': row['home_expected'], 'result': row['result_home'],
                           'goals_for': row['home_goals'], 'goals_against': row['away_goals'],
                           'venue': 'home', 'opponent': row['away_team'], 'tournament': row['tournament']})
        else:
            series.append({'date': row['date'], 'rating': row['away_rating_after'],
                           'expected': row['away_expected'], 'result': 1 - row['result_home'],
                           'goals_for': row['away_goals'], 'goals_against': row['home_goals'],
                           'venue': 'away', 'opponent': row['home_team'], 'tournament': row['tournament']})
    return pd.DataFrame(series).tail(n_last)

def elo_volatility(history_df, team, window=10):
    """Desvio padrão das mudanças de rating — mede instabilidade."""
    s = get_team_history(history_df, team)
    if len(s) < window + 1:
        return np.nan
    s['rating_change'] = s['rating'].diff()
    return s['rating_change'].tail(window).std()

def elo_momentum(history_df, team, short_window=5, long_window=15):
    """Diferença entre média curta e longa de rating — tendência de curto prazo."""
    s = get_team_history(history_df, team)
    if len(s) < long_window:
        return np.nan
    short = s['rating'].tail(short_window).mean()
    long = s['rating'].tail(long_window).mean()
    return short - long

def elo_efficiency(history_df, team, n_last=20):
    """Pontos reais vs. pontos esperados (baseado em probabilidade Elo)."""
    s = get_team_history(history_df, team).tail(n_last)
    if s.empty:
        return np.nan, np.nan
    s['points'] = s['result'].apply(lambda x: 3 if x == 1 else (1 if x == 0.5 else 0))
    s['exp_points'] = s['expected'] * 3 + 0.25 * (1 - np.abs(s['expected'] - 0.5)) * 1
    real = s['points'].mean()
    expected = s['exp_points'].mean()
    return real, expected - real

def schedule_strength(history_df, team, n_last=10):
    """Média do Elo dos adversários recentes — quanto mais forte a agenda, mais meritório o rating."""
    s = get_team_history(history_df, team).tail(n_last)
    if s.empty:
        return np.nan
    mask = ((history_df['home_team'] == team) | (history_df['away_team'] == team))
    h = history_df[mask].tail(n_last)
    opp_elos = []
    for _, row in h.iterrows():
        if row['home_team'] == team:
            opp_elos.append(row['away_rating_before'])
        else:
            opp_elos.append(row['home_rating_before'])
    return np.mean(opp_elos) if opp_elos else np.nan

def golden_window(history_df, team, min_window=8, threshold_pct=0.95):
    """Detecta períodos de pico de performance (rating acima do percentil 95)."""
    s = get_team_history(history_df, team)
    if len(s) < min_window:
        return None, None
    s['smoothed'] = s['rating'].rolling(window=5, min_periods=1, center=True).mean()
    threshold = s['smoothed'].quantile(threshold_pct)
    above = s['smoothed'] >= threshold
    s['group'] = (above != above.shift()).cumsum()
    streaks = s[above].groupby('group').size()
    if streaks.empty:
        return None, None
    best_group = streaks.idxmax()
    best = s[s['group'] == best_group]
    return best['date'].min(), best['date'].max()

def regression_speed(history_df, team):
    """Quanto tempo (em jogos) leva para regressar à média após um pico ou vale."""
    s = get_team_history(history_df, team)
    if len(s) < 20:
        return np.nan
    mean_rating = s['rating'].mean()
    s['deviation'] = s['rating'] - mean_rating
    std = s['rating'].std()
    peaks = s[s['deviation'] > 1.5 * std]
    if peaks.empty:
        return np.nan
    peak_idx = peaks.index[0]
    after = s.loc[peak_idx:]
    recovered = after[after['deviation'].abs() < 0.5 * std]
    if recovered.empty:
        return len(after)
    return recovered.index[0] - peak_idx

def h2h_psychological_edge(df, team_a, team_b, lookback_years=20):
    """Mede 'domínio psicológico': vitórias consecutivas e magnitude da superioridade."""
    cutoff = df['date'].max() - pd.Timedelta(days=365 * lookback_years)
    h2h = df[((df['home_team'] == team_a) & (df['away_team'] == team_b)) |
             ((df['home_team'] == team_b) & (df['away_team'] == team_a))]
    h2h = h2h[h2h['date'] >= cutoff].sort_values('date')
    if h2h.empty:
        return {'edge': 'Sem dados', 'streak': 0, 'dominance': 0}

    results = []
    for _, row in h2h.iterrows():
        if row['home_team'] == team_a:
            if row['home_score'] > row['away_score']:
                results.append('A')
            elif row['home_score'] == row['away_score']:
                results.append('D')
            else:
                results.append('B')
        else:
            if row['away_score'] > row['home_score']:
                results.append('A')
            elif row['away_score'] == row['home_score']:
                results.append('D')
            else:
                results.append('B')

    wins_a = results.count('A')
    wins_b = results.count('B')
    draws = results.count('D')
    total = len(results)

    streak = 0
    for r in reversed(results):
        if r == 'A':
            streak += 1
        else:
            break

    gd_sum = 0
    for _, row in h2h.iterrows():
        if row['home_team'] == team_a:
            gd_sum += row['home_score'] - row['away_score']
        else:
            gd_sum += row['away_score'] - row['home_score']
    dominance = (wins_a / total) * 0.6 + np.clip(gd_sum / (total * 3), -0.4, 0.4)

    edge = "Dominância Clara" if dominance > 0.3 else "Equilíbrio" if dominance > -0.3 else "Desvantagem"
    return {'edge': edge, 'streak': streak, 'dominance': dominance,
            'wins_a': wins_a, 'wins_b': wins_b, 'draws': draws, 'total': total,
            'gd_sum': gd_sum}

def home_advantage_by_confederation(df, history_df, confederations):
    """Calcula home advantage real por confederação."""
    results = []
    for conf, teams in confederations.items():
        conf_df = df[(df['home_team'].isin(teams)) & (df['away_team'].isin(teams)) & (df['neutral'] == False)]
        if len(conf_df) < 50:
            continue
        home_wins = (conf_df['home_score'] > conf_df['away_score']).mean()
        draws = (conf_df['home_score'] == conf_df['away_score']).mean()
        away_wins = (conf_df['home_score'] < conf_df['away_score']).mean()
        advantage = (home_wins + draws * 0.5) - 0.5
        results.append({'Confederação': conf, 'Home Win %': home_wins,
                        'Draw %': draws, 'Away Win %': away_wins,
                        'Advantage Index': advantage, 'Jogos': len(conf_df)})
    return pd.DataFrame(results)

def kelly_criterion(prob, odds, fraction=0.25):
    """Kelly Criterion fractional. f* = (bp - q) / b, onde b = odds - 1."""
    b = odds - 1
    p = prob
    q = 1 - p
    kelly = (b * p - q) / b
    return max(0, kelly) * fraction

def team_consistency(history_df, team, n_last=20):
    """Coeficiente de variação do rating — baixo = consistente, alto = imprevisível."""
    s = get_team_history(history_df, team).tail(n_last)
    if len(s) < 5:
        return np.nan
    return s['rating'].std() / s['rating'].mean()

def xg_efficiency(history_df, team, lookback_years=5):
    """Gols reais / gols esperados (Poisson) — eficiência de conversão."""
    cutoff = history_df['date'].max() - pd.Timedelta(days=365 * lookback_years)
    tdf = history_df[((history_df['home_team'] == team) | (history_df['away_team'] == team)) & (history_df['date'] >= cutoff)]
    if tdf.empty:
        return np.nan, np.nan
    real_goals = 0
    exp_goals = 0
    for _, row in tdf.iterrows():
        if row['home_team'] == team:
            real_goals += row['home_goals']
            exp_goals += row['home_expected'] * 2.5
        else:
            real_goals += row['away_goals']
            exp_goals += row['away_expected'] * 2.5
    return real_goals / max(exp_goals, 0.1), real_goals / len(tdf)
