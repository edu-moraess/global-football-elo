"""
PoissonEngine v2 — Dixon-Coles com MLE via scipy.optimize.
"""
import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import minimize
from datetime import timedelta


def _dc_tau(lam_h: float, lam_a: float, g_h: int, g_a: int, rho: float) -> float:
    if g_h == 0 and g_a == 0:
        return 1 - lam_h * lam_a * rho
    elif g_h == 1 and g_a == 0:
        return 1 + lam_a * rho
    elif g_h == 0 and g_a == 1:
        return 1 + lam_h * rho
    elif g_h == 1 and g_a == 1:
        return 1 - rho
    return 1.0


def fit_poisson_dc(df: pd.DataFrame, lookback_years: int = 10):
    cutoff = df['date'].max() - timedelta(days=365 * lookback_years)
    d = df[df['date'] >= cutoff].copy()
    d = d[d['home_score'].notna() & d['away_score'].notna()]

    teams = sorted(set(d['home_team']) | set(d['away_team']))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    hg = d['home_score'].astype(int).values
    ag = d['away_score'].astype(int).values
    hi = d['home_team'].map(idx).values
    ai = d['away_team'].map(idx).values

    avg_hg = hg.mean()
    avg_ag = ag.mean()

    def nll(params):
        att = np.exp(params[:n])
        def_ = np.exp(params[n:2*n])
        ha  = np.exp(params[2*n])
        rho = np.tanh(params[2*n + 1]) * 0.3

        lam_h = ha * att[hi] * def_[ai] * avg_hg
        lam_a =      att[ai] * def_[hi] * avg_ag

        ll = (poisson.logpmf(hg, lam_h)
            + poisson.logpmf(ag, lam_a)
            + np.log(np.maximum(1e-10, [
                _dc_tau(lam_h[k], lam_a[k], hg[k], ag[k], rho)
                for k in range(len(hg))
              ])))
        return -np.sum(ll)

    x0 = np.zeros(2 * n + 2)
    res = minimize(nll, x0, method='L-BFGS-B',
                   options={'maxiter': 300, 'ftol': 1e-8})

    att  = np.exp(res.x[:n]);       att  /= att.mean()
    def_ = np.exp(res.x[n:2*n]);    def_ /= def_.mean()
    ha   = np.exp(res.x[2*n])
    rho  = float(np.tanh(res.x[2*n + 1]) * 0.3)

    attack_d  = {t: att[i]  for t, i in idx.items()}
    defense_d = {t: def_[i] for t, i in idx.items()}
    league_avg = (avg_hg, avg_ag)

    return attack_d, defense_d, ha, rho, league_avg


def predict_match(home: str, away: str, neutral: bool,
                  attack: dict, defense: dict,
                  home_adv: float, rho: float,
                  league_avg: tuple, elo_ratings: dict,
                  max_goals: int = 10) -> dict:
    att_h  = attack.get(home,  1.0)
    def_h  = defense.get(home, 1.0)
    att_a  = attack.get(away,  1.0)
    def_a  = defense.get(away, 1.0)
    avg_h, avg_a = league_avg

    ha_factor = 1.0 if neutral else home_adv
    lam_h = ha_factor * att_h * def_a * avg_h
    lam_a =             att_a * def_h * avg_a

    elo_h = elo_ratings.get(home, 1500)
    elo_a = elo_ratings.get(away, 1500)
    elo_adj = np.exp((elo_h - elo_a) / 2500)
    lam_h *= elo_adj ** 0.5
    lam_a /= elo_adj ** 0.5

    pmf_h = poisson.pmf(np.arange(max_goals), lam_h)
    pmf_a = poisson.pmf(np.arange(max_goals), lam_a)
    M = np.outer(pmf_h, pmf_a)

    for i in range(2):
        for j in range(2):
            M[i, j] *= max(1e-10, _dc_tau(lam_h, lam_a, i, j, rho))

    M = np.clip(M, 0, None)
    M /= M.sum()

    p_home = float(np.sum(np.tril(M, -1)))
    p_draw = float(np.sum(np.diag(M)))
    p_away = float(np.sum(np.triu(M, 1)))

    g_idx = np.arange(max_goals)
    xg_h = float(np.sum(g_idx[:, None] * M))
    xg_a = float(np.sum(g_idx[None, :] * M))

    return dict(
        lam_h=lam_h, lam_a=lam_a,
        xg_h=xg_h, xg_a=xg_a,
        p_home=p_home, p_draw=p_draw, p_away=p_away,
        matrix=M,
        elo_h=elo_h, elo_a=elo_a,
        supremacy=xg_h - xg_a,
        home=home, away=away,
    )


def monte_carlo_match(lam_h: float, lam_a: float,
                      iterations: int = 30_000, rho: float = -0.08) -> tuple:
    rng = np.random.default_rng(42)
    gh = rng.poisson(lam_h, iterations)
    ga = rng.poisson(lam_a, iterations)

    low = (gh <= 1) & (ga <= 1)
    for k in np.where(low)[0]:
        tau = _dc_tau(lam_h, lam_a, gh[k], ga[k], rho)
        if rng.random() > tau:
            gh[k] = rng.poisson(lam_h)
            ga[k] = rng.poisson(lam_a)

    p_h = (gh > ga).mean()
    p_d = (gh == ga).mean()
    p_a = (gh < ga).mean()
    return float(p_h), float(p_d), float(p_a)


def score_distribution(M: np.ndarray, top_n: int = 10) -> list[tuple[str, float]]:
    scores = []
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            scores.append((f"{i}–{j}", float(M[i, j])))
    return sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]