"""
Motor Poisson-Dixon-Coles para predição de placares.
O futebol não é Poisson puro — empates de baixo placar são mais frequentes.
O ajuste ρ (rho) captura essa dependência.
"""
import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import minimize


def get_poisson_strengths(df, lookback_years):
    """
    Estima forças de ataque/defesa por time via MLE Dixon-Coles simplificado.
    Retorna: (attack dict, defense dict, league_avg)
    """
    from datetime import timedelta
    cutoff = df['date'].max() - timedelta(days=365 * lookback_years)
    df = df[df['date'] >= cutoff].copy()

    teams = pd.concat([df['home_team'], df['away_team']]).unique()
    n = len(teams)
    team_idx = {t: i for i, t in enumerate(teams)}

    home_goals = df['home_score'].values
    away_goals = df['away_score'].values
    home_idx = df['home_team'].map(team_idx).values
    away_idx = df['away_team'].map(team_idx).values

    league_avg_home = home_goals.mean()
    league_avg_away = away_goals.mean()

    attack = np.ones(n)
    defense = np.ones(n)
    home_adv = 1.0

    def nll(params):
        att = params[:n]
        def_ = params[n:2*n]
        ha = params[2*n]
        att = att / att.mean()
        def_ = def_ / def_.mean()
        lam_h = ha * att[home_idx] * def_[away_idx] * league_avg_home
        lam_a = att[away_idx] * def_[home_idx] * league_avg_away
        ll = poisson.logpmf(home_goals, lam_h) + poisson.logpmf(away_goals, lam_a)
        return -np.sum(ll)

    x0 = np.concatenate([attack, defense, [home_adv]])
    bounds = [(0.1, 5.0)] * (2 * n) + [(0.5, 2.0)]
    result = minimize(nll, x0, bounds=bounds, method='L-BFGS-B', options={'maxiter': 100})

    att = result.x[:n] / result.x[:n].mean()
    def_ = result.x[n:2*n] / result.x[n:2*n].mean()
    ha = result.x[2*n]

    attack_dict = {t: att[i] for t, i in team_idx.items()}
    defense_dict = {t: def_[i] for t, i in team_idx.items()}

    return attack_dict, defense_dict, (league_avg_home, league_avg_away)


def dixon_coles_adjustment(lambda1, lambda2, rho=-0.08):
    """Ajuste de dependência Dixon-Coles para empates de baixo placar."""
    if rho == 0:
        return 1.0
    return 1 + rho * (lambda1 * lambda2) if lambda1 * lambda2 < 0.5 else 1.0


def predict_match(home_team, away_team, neutral, attack, defense, league_avg, elo_ratings, rho=-0.08):
    """
    Prediz um confronto. Retorna dict com probabilidades, xG, matrix, etc.
    """
    elo_h = elo_ratings.get(home_team, 1500)
    elo_a = elo_ratings.get(away_team, 1500)

    att_h = attack.get(home_team, 1.0)
    def_h = defense.get(home_team, 1.0)
    att_a = attack.get(away_team, 1.0)
    def_a = defense.get(away_team, 1.0)

    l_avg_h, l_avg_a = league_avg
    home_adv = 1.0 if neutral else 1.35

    lambda_h = home_adv * att_h * def_a * l_avg_h
    lambda_a = att_a * def_h * l_avg_a

    elo_diff = elo_h - elo_a
    elo_adj = 1 + (elo_diff / 4000)
    lambda_h *= elo_adj
    lambda_a /= elo_adj

    max_goals = 10
    matrix = np.zeros((max_goals, max_goals))
    for i in range(max_goals):
        for j in range(max_goals):
            p_h = poisson.pmf(i, lambda_h)
            p_a = poisson.pmf(j, lambda_a)
            adj = dixon_coles_adjustment(lambda_h, lambda_a, rho) if i <= 1 and j <= 1 else 1.0
            matrix[i, j] = p_h * p_a * adj

    matrix /= matrix.sum()

    p_home = np.sum(np.tril(matrix, -1))
    p_draw = np.sum(np.diag(matrix))
    p_away = np.sum(np.triu(matrix, 1))

    total = p_home + p_draw + p_away
    p_home /= total
    p_draw /= total
    p_away /= total

    xg_h = np.sum([i * np.sum(matrix[i, :]) for i in range(max_goals)])
    xg_a = np.sum([j * np.sum(matrix[:, j]) for j in range(max_goals)])

    return {
        'elo_home': elo_h, 'elo_away': elo_a,
        'l_home': lambda_h, 'l_away': lambda_a,
        'xg_home': xg_h, 'xg_away': xg_a,
        'p_home': p_home, 'p_draw': p_draw, 'p_away': p_away,
        'matrix': matrix,
        'goal_supremacy': xg_h - xg_a
    }


def monte_carlo(lambda_h, lambda_a, iterations=20000, rho=-0.08):
    """
    Simula Monte Carlo para obter probabilidades empíricas.
    """
    np.random.seed(42)
    goals_h = np.random.poisson(lambda_h, iterations)
    goals_a = np.random.poisson(lambda_a, iterations)

    if rho != 0:
        for idx in range(iterations):
            gh, ga = goals_h[idx], goals_a[idx]
            if gh <= 1 and ga <= 1:
                adj = 1 + rho * lambda_h * lambda_a
                if np.random.random() > adj:
                    goals_h[idx] = np.random.poisson(lambda_h)
                    goals_a[idx] = np.random.poisson(lambda_a)

    wins_h = np.sum(goals_h > goals_a)
    draws = np.sum(goals_h == goals_a)
    wins_a = np.sum(goals_h < goals_a)

    return wins_h / iterations, draws / iterations, wins_a / iterations
