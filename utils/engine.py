# utils/engine.py — Motor Elo-Poisson Quant Restaurado & Otimizado
import pandas as pd
import numpy as np
from scipy.stats import poisson
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

INITIAL_ELO = 1500
K_FACTOR_BASE = 32
ELO_SCALE = 400
HOME_ADVANTAGE = 55

TOURNAMENT_WEIGHTS = {
    'FIFA World Cup': 60, 'Copa América': 50, 'UEFA Euro': 50,
    'FIFA World Cup qualification': 40, 'UEFA Euro qualification': 35,
    'CONMEBOL qualification': 35, 'AFC Asian Cup': 30,
    'Africa Cup of Nations': 30, 'CONCACAF Gold Cup': 30,
    'OFC Nations Cup': 25, 'Friendly': 20,
}

CONFEDERATIONS = {
    'UEFA': ['Albania', 'Andorra', 'Armenia', 'Austria', 'Azerbaijan', 'Belarus', 'Belgium',
             'Bosnia and Herzegovina', 'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic',
             'Denmark', 'England', 'Estonia', 'Faroe Islands', 'Finland', 'France', 'Georgia',
             'Germany', 'Gibraltar', 'Greece', 'Hungary', 'Iceland', 'Israel', 'Italy',
             'Kazakhstan', 'Kosovo', 'Latvia', 'Liechtenstein', 'Lithuania', 'Luxembourg',
             'Malta', 'Moldova', 'Monaco', 'Montenegro', 'Netherlands', 'North Macedonia',
             'Northern Ireland', 'Norway', 'Poland', 'Portugal', 'Republic of Ireland',
             'Romania', 'Russia', 'San Marino', 'Scotland', 'Serbia', 'Slovakia', 'Slovenia',
             'Spain', 'Sweden', 'Switzerland', 'Turkey', 'Ukraine', 'Wales'],
    'CONMEBOL': ['Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'Paraguay',
                 'Peru', 'Uruguay', 'Venezuela'],
    'CAF': ['Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cameroon',
            'Cape Verde', 'Central African Republic', 'Chad', 'Comoros', 'Congo', 'DR Congo',
            'Djibouti', 'Egypt', 'Equatorial Guinea', 'Eritrea', 'Eswatini', 'Ethiopia',
            'Gabon', 'Gambia', 'Ghana', 'Guinea', 'Guinea-Bissau', 'Ivory Coast', 'Kenya',
            'Lesotho', 'Liberia', 'Libya', 'Madagascar', 'Malawi', 'Mali', 'Mauritania',
            'Mauritius', 'Morocco', 'Mozambique', 'Namibia', 'Niger', 'Nigeria', 'Rwanda',
            'São Tomé and Príncipe', 'Senegal', 'Seychelles', 'Sierra Leone', 'Somalia',
            'South Africa', 'South Sudan', 'Sudan', 'Tanzania', 'Togo', 'Tunisia', 'Uganda',
            'Zambia', 'Zimbabwe'],
    'AFC': ['Afghanistan', 'Australia', 'Bahrain', 'Bangladesh', 'Bhutan', 'Brunei', 'Cambodia',
            'China', 'Chinese Taipei', 'Guam', 'Hong Kong', 'India', 'Indonesia', 'Iran',
            'Iraq', 'Japan', 'Jordan', 'Kuwait', 'Kyrgyzstan', 'Laos', 'Lebanon', 'Macau',
            'Malaysia', 'Maldives', 'Mongolia', 'Myanmar', 'Nepal', 'North Korea', 'Oman',
            'Pakistan', 'Palestine', 'Philippines', 'Qatar', 'Saudi Arabia', 'Singapore',
            'South Korea', 'Sri Lanka', 'Syria', 'Tajikistan', 'Thailand', 'Timor-Leste',
            'Turkmenistan', 'United Arab Emirates', 'Uzbekistan', 'Vietnam', 'Yemen'],
    'CONCACAF': ['Antigua and Barbuda', 'Bahamas', 'Barbados', 'Belize', 'Bermuda', 'Canada',
                 'Costa Rica', 'Cuba', 'Dominica', 'Dominican Republic', 'El Salvador',
                 'Grenada', 'Guatemala', 'Haiti', 'Honduras', 'Jamaica', 'Mexico',
                 'Nicaragua', 'Panama', 'Saint Kitts and Nevis', 'Saint Lucia',
                 'Saint Vincent and the Grenadines', 'Trinidad and Tobago', 'United States'],
    'OFC': ['American Samoa', 'Cook Islands', 'Fiji', 'New Caledonia', 'New Zealand',
            'Papua New Guinea', 'Samoa', 'Solomon Islands', 'Tahiti', 'Tonga', 'Vanuatu'],
}

def get_confederation(team):
    for conf, teams in CONFEDERATIONS.items():
        if team in teams:
            return conf
    return 'Other'

def goal_diff_multiplier(goal_diff):
    if goal_diff <= 1:
        return 1.0
    elif goal_diff == 2:
        return 1.5
    else:
        return (11 + goal_diff) / 8.0

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

        if home_goals > away_goals:
            res_home = 1.0
        elif home_goals == away_goals:
            res_home = 0.5
        else:
            res_home = 0.0

        weight = TOURNAMENT_WEIGHTS.get(tournament, 20)
        gd = abs(float(home_goals) - float(away_goals))
        k = K_FACTOR_BASE * (weight / 30.0) * goal_diff_multiplier(gd)

        ratings[home] = r_home + k * (res_home - exp_home)
        ratings[away] = r_away + k * ((1 - res_home) - (1 - exp_home))

        history.append({
            'date': row['date'],
            'home_team': home,
            'away_team': away,
            'home_goals': home_goals,
            'away_goals': away_goals,
            'tournament': tournament,
            'neutral': neutral,
            'home_rating_before': r_home,
            'away_rating_before': r_away,
            'home_rating_after': ratings[home],
            'away_rating_after': ratings[away],
            'home_expected': exp_home,
            'away_expected': 1 - exp_home,
            'k_factor': k,
            'goal_diff': gd,
            'result_home': res_home,
        })

    return ratings, pd.DataFrame(history)

def get_poisson_strengths(df, lookback_years):
    cutoff = df['date'].max() - timedelta(days=365 * lookback_years)
    df_f = df[df['date'] >= cutoff].copy()

    home_matches = df_f.groupby('home_team').size()
    away_matches = df_f.groupby('away_team').size()
    matches = home_matches.add(away_matches, fill_value=0)

    home_goals = df_f.groupby('home_team')['home_score'].sum()
    away_goals = df_f.groupby('away_team')['away_score'].sum()
    goals_scored = home_goals.add(away_goals, fill_value=0)
    avg_scored = (goals_scored / matches).fillna(0)

    home_conceded = df_f.groupby('home_team')['away_score'].sum()
    away_conceded = df_f.groupby('away_team')['home_score'].sum()
    goals_conceded = home_conceded.add(away_conceded, fill_value=0)
    avg_conceded = (goals_conceded / matches).fillna(0)

    league_avg = (df_f['home_score'].sum() + df_f['away_score'].sum()) / (2 * len(df_f))

    attack = avg_scored / league_avg
    defense = avg_conceded / league_avg
    return attack, defense, league_avg

def predict_match(home, away, neutral, attack, defense, league_avg, elo_ratings, rho=-0.08):
    """Poisson com correlação Dixon-Coles (rho) para empates mais realistas."""
    att_h, def_h = attack.get(home, 1.0), defense.get(home, 1.0)
    att_a, def_a = attack.get(away, 1.0), defense.get(away, 1.0)

    l_home = league_avg * att_h * def_a * (1.12 if not neutral else 1.0)
    l_away = league_avg * att_a * def_h * (0.88 if not neutral else 1.0)

    max_g = 10
    pmf_h = poisson.pmf(np.arange(max_g + 1), l_home)
    pmf_a = poisson.pmf(np.arange(max_g + 1), l_away)
    prob_matrix = np.outer(pmf_h, pmf_a)

    for i in range(max_g + 1):
        for j in range(max_g + 1):
            if i == 0 and j == 0:
                prob_matrix[i, j] *= (1 + rho)
            elif i == 1 and j == 0:
                prob_matrix[i, j] *= (1 - rho * 0.5)
            elif i == 0 and j == 1:
                prob_matrix[i, j] *= (1 - rho * 0.5)
            elif i == 1 and j == 1:
                prob_matrix[i, j] *= (1 + rho * 0.3)

    prob_matrix /= prob_matrix.sum()

    p_home = np.sum(np.tril(prob_matrix, -1))
    p_draw = np.sum(np.diag(prob_matrix))
    p_away = np.sum(np.triu(prob_matrix, 1))

    goal_supremacy = sum((i - j) * prob_matrix[i, j] for i in range(max_g + 1) for j in range(max_g + 1))
    xg_home = sum(i * prob_matrix[i, j] for i in range(max_g + 1) for j in range(max_g + 1))
    xg_away = sum(j * prob_matrix[i, j] for i in range(max_g + 1) for j in range(max_g + 1))

    return {
        "l_home": l_home,
        "l_away": l_away,
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "matrix": prob_matrix,
        "elo_home": elo_ratings.get(home, INITIAL_ELO),
        "elo_away": elo_ratings.get(away, INITIAL_ELO),
        "goal_supremacy": goal_supremacy,
        "xg_home": xg_home,
        "xg_away": xg_away,
    }

def monte_carlo(l_home, l_away, iterations=20000):
    h_goals = np.random.poisson(l_home, iterations)
    a_goals = np.random.poisson(l_away, iterations)
    p_h = np.sum(h_goals > a_goals) / iterations
    p_d = np.sum(h_goals == a_goals) / iterations
    p_a = np.sum(h_goals < a_goals) / iterations
    return p_h, p_d, p_a
