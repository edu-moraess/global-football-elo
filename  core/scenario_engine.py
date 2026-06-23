"""
Análise de cenários e sensibilidade para predições.
Foco em entender incerteza e robustez do modelo — sem relação com apostas.
"""
import numpy as np
import pandas as pd
from .poisson_engine import predict_match


def sensitivity_analysis(home_team, away_team, neutral, attack, defense, league_avg, elo_ratings,
                         elo_shift_range=(-200, 200), n_points=9):
    """
    Analisa como a probabilidade de vitória muda com variações no Elo.
    Útil para entender sensibilidade do modelo a incertezas de rating.
    """
    shifts = np.linspace(elo_shift_range[0], elo_shift_range[1], n_points)
    results = []

    for shift in shifts:
        modified_elo = elo_ratings.copy()
        modified_elo[home_team] = modified_elo.get(home_team, 1500) + shift

        res = predict_match(home_team, away_team, neutral, attack, defense, league_avg, modified_elo, rho=-0.08)
        results.append({
            'Elo Shift': shift,
            'Prob Casa': res['p_home'],
            'Prob Empate': res['p_draw'],
            'Prob Fora': res['p_away'],
            'xG Casa': res['xg_home'],
            'xG Fora': res['xg_away'],
            'Supremacy': res['goal_supremacy']
        })

    return pd.DataFrame(results)


def scenario_matrix(home_team, away_team, attack, defense, league_avg, elo_ratings,
                    scenarios=None):
    """
    Matriz de cenários: compara diferentes contextos (neutro, casa, desfalques, etc).
    """
    if scenarios is None:
        scenarios = {
            'Campo Neutro': {'neutral': True, 'elo_shift_h': 0, 'elo_shift_a': 0},
            'Casa Normal': {'neutral': False, 'elo_shift_h': 0, 'elo_shift_a': 0},
            'Casa + Pressão (Elo +50)': {'neutral': False, 'elo_shift_h': 50, 'elo_shift_a': 0},
            'Casa - Lesões (Elo -50)': {'neutral': False, 'elo_shift_h': -50, 'elo_shift_a': 0},
            'Fora + Ascensão (Elo +50)': {'neutral': False, 'elo_shift_h': 0, 'elo_shift_a': 50},
        }

    results = []
    for name, params in scenarios.items():
        modified_elo = elo_ratings.copy()
        modified_elo[home_team] = modified_elo.get(home_team, 1500) + params['elo_shift_h']
        modified_elo[away_team] = modified_elo.get(away_team, 1500) + params['elo_shift_a']

        res = predict_match(home_team, away_team, params['neutral'], attack, defense, league_avg, modified_elo, rho=-0.08)
        results.append({
            'Cenário': name,
            'Prob Casa': res['p_home'],
            'Prob Empate': res['p_draw'],
            'Prob Fora': res['p_away'],
            'xG Casa': res['xg_home'],
            'xG Fora': res['xg_away'],
            'Supremacy': res['goal_supremacy']
        })

    return pd.DataFrame(results)


def what_if_elo_shift(elo_ratings, team, shift, all_teams):
    """
    Simula o impacto de uma mudança de Elo de um time no ranking global.
    Útil para entender como um torneio ou jogo pode alterar o cenário global.
    """
    modified = elo_ratings.copy()
    modified[team] = modified.get(team, 1500) + shift

    ranking = sorted(modified.items(), key=lambda x: x[1], reverse=True)
    original_rank = {t: i+1 for i, (t, _) in enumerate(sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True))}
    new_rank = {t: i+1 for i, (t, _) in enumerate(ranking)}

    impact = []
    for t in all_teams:
        if t in original_rank and t in new_rank:
            impact.append({
                'Seleção': t,
                'Elo Original': elo_ratings.get(t, 1500),
                'Elo Novo': modified.get(t, 1500),
                'Posição Original': original_rank[t],
                'Posição Nova': new_rank[t],
                'Δ Posição': original_rank[t] - new_rank[t]
            })

    return pd.DataFrame(impact).sort_values('Posição Nova')
