"""
Edumetria | Football Elo Rating Engine
Calcula ratings Elo históricos para seleções nacionais (estilo World Football Elo Ratings)
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

# Peso por importância da competição (estilo eloratings.net)
TOURNAMENT_WEIGHTS = {
    "FIFA World Cup": 60,
    "FIFA World Cup qualification": 40,
    "Copa América": 50,
    "UEFA Euro": 50,
    "UEFA Euro qualification": 35,
    "African Cup of Nations": 50,
    "African Cup of Nations qualification": 35,
    "AFC Asian Cup": 50,
    "AFC Asian Cup qualification": 35,
    "CONCACAF Nations League": 35,
    "UEFA Nations League": 35,
    "Friendly": 20,
}
DEFAULT_WEIGHT = 30
K_FACTOR_BASE = 32
INITIAL_ELO = 1500


def load_former_names():
    fn = pd.read_csv(DATA_DIR / "former_names.csv")
    return fn


def build_name_map(former_names: pd.DataFrame, as_of_date=None):
    """Mapeia nomes antigos -> nome atual. Usado para normalizar séries históricas."""
    mapping = {}
    for _, row in former_names.iterrows():
        mapping[row["former"]] = row["current"]
    return mapping


def normalize_team_names(df: pd.DataFrame, name_map: dict, cols=("home_team", "away_team")):
    df = df.copy()
    for c in cols:
        df[c] = df[c].replace(name_map)
    return df


def expected_score(rating_a, rating_b):
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def goal_diff_multiplier(goal_diff):
    """Multiplicador de K em função da diferença de gols (estilo eloratings.net)."""
    if goal_diff <= 1:
        return 1.0
    elif goal_diff == 2:
        return 1.5
    else:
        return (11 + goal_diff) / 8.0


def compute_elo_history(results: pd.DataFrame, name_map: dict):
    """
    Retorna:
      - ratings_history: DataFrame long (date, team, elo)
      - current_ratings: dict {team: elo}
      - matches_enriched: results com elo_home_pre, elo_away_pre, prob_home, k_used
    """
    df = results.copy()
    df = normalize_team_names(df, name_map)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=["home_score", "away_score"])

    ratings = {}
    history_rows = []
    enriched_rows = []

    for _, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        hs, as_ = row["home_score"], row["away_score"]
        tournament = row["tournament"]
        neutral = row.get("neutral", False)

        r_home = ratings.get(home, INITIAL_ELO)
        r_away = ratings.get(away, INITIAL_ELO)

        # Vantagem de campo (~50 pts) se não for neutro
        adj_home = r_home if neutral else r_home + 50
        adj_away = r_away

        exp_home = expected_score(adj_home, adj_away)
        exp_away = 1 - exp_home

        if hs > as_:
            score_home, score_away = 1.0, 0.0
        elif hs < as_:
            score_home, score_away = 0.0, 1.0
        else:
            score_home, score_away = 0.5, 0.5

        weight = TOURNAMENT_WEIGHTS.get(tournament, DEFAULT_WEIGHT)
        gd_mult = goal_diff_multiplier(abs(hs - as_))
        k = (K_FACTOR_BASE * weight / 30.0) * gd_mult

        new_r_home = r_home + k * (score_home - exp_home)
        new_r_away = r_away + k * (score_away - exp_away)

        ratings[home] = new_r_home
        ratings[away] = new_r_away

        history_rows.append({"date": row["date"], "team": home, "elo": new_r_home})
        history_rows.append({"date": row["date"], "team": away, "elo": new_r_away})

        enriched_rows.append({
            "date": row["date"], "home_team": home, "away_team": away,
            "home_score": hs, "away_score": as_, "tournament": tournament,
            "elo_home_pre": r_home, "elo_away_pre": r_away,
            "prob_home_pre": exp_home, "prob_away_pre": exp_away,
            "k_used": k,
        })

    history = pd.DataFrame(history_rows)
    enriched = pd.DataFrame(enriched_rows)
    return history, ratings, enriched


def get_elo_timeseries(history: pd.DataFrame, team: str):
    sub = history[history["team"] == team].copy()
    sub = sub.sort_values("date")
    return sub


def predict_match(rating_a, rating_b, neutral=True):
    adj_a = rating_a if neutral else rating_a + 50
    p_a = expected_score(adj_a, rating_b)
    return p_a, 1 - p_a


# ---- Poisson goal model ----
def team_attack_defense_strength(results: pd.DataFrame, name_map: dict, lookback_years=None):
    """Calcula força ofensiva/defensiva relativa (estilo Dixon-Coles simplificado)."""
    df = normalize_team_names(results.copy(), name_map)
    df["date"] = pd.to_datetime(df["date"])
    if lookback_years:
        cutoff = df["date"].max() - pd.DateOffset(years=lookback_years)
        df = df[df["date"] >= cutoff]
    df = df.dropna(subset=["home_score", "away_score"])

    avg_home_goals = df["home_score"].mean()
    avg_away_goals = df["away_score"].mean()

    home_goals = df.groupby("home_team")["home_score"].agg(["mean", "count"])
    home_conceded = df.groupby("home_team")["away_score"].mean()
    away_goals = df.groupby("away_team")["away_score"].agg(["mean", "count"])
    away_conceded = df.groupby("away_team")["home_score"].mean()

    teams = set(home_goals.index) | set(away_goals.index)
    strength = {}
    for t in teams:
        n_home = home_goals["count"].get(t, 0)
        n_away = away_goals["count"].get(t, 0)
        if n_home + n_away < 5:
            continue
        atk_home = home_goals["mean"].get(t, avg_home_goals) / avg_home_goals
        atk_away = away_goals["mean"].get(t, avg_away_goals) / avg_away_goals
        def_home = home_conceded.get(t, avg_away_goals) / avg_away_goals
        def_away = away_conceded.get(t, avg_home_goals) / avg_home_goals

        attack = np.average([atk_home, atk_away], weights=[n_home or 1, n_away or 1])
        defense = np.average([def_home, def_away], weights=[n_home or 1, n_away or 1])
        strength[t] = {"attack": attack, "defense": defense, "n_matches": n_home + n_away}

    return strength, avg_home_goals, avg_away_goals


def poisson_match_probs(home_team, away_team, strength, avg_home_goals, avg_away_goals, max_goals=8):
    from scipy.stats import poisson

    h = strength.get(home_team, {"attack": 1.0, "defense": 1.0})
    a = strength.get(away_team, {"attack": 1.0, "defense": 1.0})

    lam_home = avg_home_goals * h["attack"] * a["defense"]
    lam_away = avg_away_goals * a["attack"] * h["defense"]

    lam_home = max(lam_home, 0.05)
    lam_away = max(lam_away, 0.05)

    home_probs = [poisson.pmf(i, lam_home) for i in range(max_goals + 1)]
    away_probs = [poisson.pmf(i, lam_away) for i in range(max_goals + 1)]

    score_matrix = np.outer(home_probs, away_probs)

    p_home_win = np.tril(score_matrix, -1).sum()
    p_draw = np.trace(score_matrix)
    p_away_win = np.triu(score_matrix, 1).sum()

    return {
        "lambda_home": lam_home,
        "lambda_away": lam_away,
        "p_home_win": p_home_win,
        "p_draw": p_draw,
        "p_away_win": p_away_win,
        "score_matrix": score_matrix,
    }
