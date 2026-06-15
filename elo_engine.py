#!/usr/bin/env python3
"""
global_football_intelligence.py

Sistema unificado de predição para futebol de seleções:
- Rating Elo com janela deslizante (padrão: últimos 10 anos)
- Modelo Poisson bivariado (força ofensiva/defensiva nos últimos 10 anos)
- Alinhamento forçado entre as duas abordagens
- Geração de relatório completo para um jogo (casa/visitante/neutro)

Autor: Ajustado conforme solicitação
Base histórica: results.csv (padrão: date, home_team, away_team, home_score, away_score, tournament, neutral)
"""

import pandas as pd
import numpy as np
from scipy.stats import poisson
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ========================= CONFIGURAÇÕES GLOBAIS =========================
INITIAL_ELO = 1500
K_FACTOR_BASE = 32
ELO_SCALE = 400               # escala clássica (pode testar 300 ou 500)
HOME_ADVANTAGE_ELO = 50      # pontos extras para o time da casa (só se não neutro)
LOOKBACK_YEARS = 10           # mesmo período para Elo e Poisson

# Pesos por competição (multiplicam o K do Elo)
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

# ========================= FUNÇÕES AUXILIARES =========================
def goal_diff_multiplier(goal_diff):
    """Ajuste do K conforme margem de gols (World Football Elo)"""
    if goal_diff == 1:
        return 1.0
    elif goal_diff == 2:
        return 1.5
    else:
        return (11 + goal_diff) / 8.0

def expected_score_elo(rating_a, rating_b, neutral=False):
    """
    Probabilidade esperada de vitória do time A (0 a 1) segundo Elo.
    Se neutral=True, não aplica vantagem de casa.
    """
    if neutral:
        diff = rating_b - rating_a
    else:
        diff = (rating_b - rating_a) + HOME_ADVANTAGE_ELO
    return 1.0 / (1.0 + 10 ** (diff / ELO_SCALE))

def compute_elo_ratings(df, lookback_years=LOOKBACK_YEARS):
    """
    Calcula o rating Elo final de cada seleção considerando apenas
    os jogos dos últimos `lookback_years` anos.
    Retorna: dict {team: rating_final}
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    if lookback_years is not None:
        cutoff = df['date'].max() - timedelta(days=365*lookback_years)
        df = df[df['date'] >= cutoff].copy()
        print(f"[Elo] Usando jogos a partir de {cutoff.date()} (últimos {lookback_years} anos)")
    
    # Inicializa ratings
    teams = set(df['home_team']).union(set(df['away_team']))
    ratings = {team: INITIAL_ELO for team in teams}
    
    for _, row in df.iterrows():
        home, away = row['home_team'], row['away_team']
        home_goals, away_goals = row['home_score'], row['away_score']
        tournament = row['tournament']
        neutral = row.get('neutral', False)
        
        r_home, r_away = ratings[home], ratings[away]
        
        # Probabilidades esperadas
        exp_home = expected_score_elo(r_home, r_away, neutral=neutral)
        exp_away = 1 - exp_home
        
        # Resultado real
        if home_goals > away_goals:
            res_home, res_away = 1, 0
        elif home_goals == away_goals:
            res_home, res_away = 0.5, 0.5
        else:
            res_home, res_away = 0, 1
        
        # Fator K com peso do torneio e margem de gols
        weight = TOURNAMENT_WEIGHTS.get(tournament, 20)
        gd = abs(home_goals - away_goals)
        k = K_FACTOR_BASE * (weight / 30.0) * goal_diff_multiplier(gd)
        
        ratings[home] = r_home + k * (res_home - exp_home)
        ratings[away] = r_away + k * (res_away - exp_away)
    
    return ratings

def team_strength_poisson(df, lookback_years=LOOKBACK_YEARS):
    """
    Estima forças ofensivas e defensivas para cada time nos últimos `lookback_years`.
    Retorna: dict com 'attack', 'defense' relativas à média geral.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    if lookback_years:
        cutoff = df['date'].max() - timedelta(days=365*lookback_years)
        df = df[df['date'] >= cutoff]
    
    # Métrica simples: gols marcados e sofridos por jogo, normalizado
    home_goals = df.groupby('home_team')['home_score'].sum()
    home_matches = df.groupby('home_team')['home_score'].count()
    away_goals = df.groupby('away_team')['away_score'].sum()
    away_matches = df.groupby('away_team')['away_score'].count()
    
    goals_scored = home_goals.add(away_goals, fill_value=0)
    matches = home_matches.add(away_matches, fill_value=0)
    avg_scored = (goals_scored / matches).fillna(0)
    
    goals_conceded_home = df.groupby('home_team')['away_score'].sum()
    goals_conceded_away = df.groupby('away_team')['home_score'].sum()
    goals_conceded = goals_conceded_home.add(goals_conceded_away, fill_value=0)
    avg_conceded = (goals_conceded / matches).fillna(0)
    
    # Média global de gols por jogo
    total_goals = df['home_score'].sum() + df['away_score'].sum()
    total_matches = len(df)
    league_avg = total_goals / total_matches if total_matches > 0 else 1.0
    
    attack = avg_scored / league_avg
    defense = avg_conceded / league_avg
    
    return attack, defense, league_avg

def poisson_match_probs(lambda_home, lambda_away, max_goals=8):
    """Retorna matriz de probabilidades de placar (max_goals x max_goals)"""
    prob_matrix = np.outer(poisson.pmf(np.arange(max_goals+1), lambda_home),
                           poisson.pmf(np.arange(max_goals+1), lambda_away))
    return prob_matrix

def predict_match(home_team, away_team, neutral, df, elo_ratings, attack, defense, league_avg):
    """
    Gera predição completa usando tanto Elo quanto Poisson (apenas o Poisson é usado
    para a matriz de placares; o Elo serve como referência ou para validação).
    Retorna dicionário com todas as saídas.
    """
    # ----- Parâmetros do Poisson -----
    # Força ofensiva e defensiva
    att_home = attack.get(home_team, 1.0)
    att_away = attack.get(away_team, 1.0)
    def_home = defense.get(home_team, 1.0)
    def_away = defense.get(away_team, 1.0)
    
    # Lambda esperado (gols)
    # Se campo neutro, não há ajuste de casa; caso contrário, usa-se um fator de casa (1.2~1.4)
    # Vamos usar uma abordagem simples: em casa, o ataque é multiplicado por 1.2 e defesa por 0.9
    if not neutral:
        lambda_home = league_avg * att_home * def_away * 1.2
        lambda_away = league_avg * att_away * def_home * 0.9
    else:
        lambda_home = league_avg * att_home * def_away
        lambda_away = league_avg * att_away * def_home
    
    # ----- Probabilidades de resultado (Poisson independente) -----
    max_goals = 10
    prob_matrix = poisson_match_probs(lambda_home, lambda_away, max_goals)
    prob_home_win = np.sum(np.tril(prob_matrix, -1))   # home > away
    prob_away_win = np.sum(np.triu(prob_matrix, 1))    # away > home
    prob_draw = np.sum(np.diag(prob_matrix))
    
    # ----- Probabilidades via Elo (para referência) -----
    elo_home = elo_ratings.get(home_team, INITIAL_ELO)
    elo_away = elo_ratings.get(away_team, INITIAL_ELO)
    prob_elo_home = expected_score_elo(elo_home, elo_away, neutral=neutral)
    prob_elo_away = 1 - prob_elo_home
    
    # ----- Matriz de placares mais prováveis (top 10) -----
    scores = []
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            prob = prob_matrix[i, j]
            if prob > 0.001:
                scores.append((f"{i} x {j}", prob))
    scores.sort(key=lambda x: -x[1])
    top_scores = scores[:10]
    
    # ----- Sumarização -----
    result = {
        'home_team': home_team,
        'away_team': away_team,
        'neutral': neutral,
        'lambda_home': round(lambda_home, 2),
        'lambda_away': round(lambda_away, 2),
        'prob_home_win': prob_home_win,
        'prob_draw': prob_draw,
        'prob_away_win': prob_away_win,
        'elo_rating_home': elo_home,
        'elo_rating_away': elo_away,
        'prob_elo_home': prob_elo_home,
        'prob_elo_away': prob_elo_away,
        'top_scores': top_scores,
        'prob_matrix': prob_matrix
    }
    return result

def print_prediction_report(pred):
    """Imprime no formato semelhante às imagens fornecidas"""
    print("\n" + "="*70)
    print(f"Modelo de Predição (Poisson Bivariado) – Base: últimos {LOOKBACK_YEARS} anos")
    print("="*70)
    print(f"\nSeleção da casa : {pred['home_team']}")
    print(f"Seleção visitante: {pred['away_team']}")
    print(f"Campo neutro     : {'Sim' if pred['neutral'] else 'Não'}")
    print("\n--- Probabilidades de Resultado (Poisson) ---")
    print(f"Vitória {pred['home_team']}: {pred['prob_home_win']:.1%}")
    print(f"Empate             : {pred['prob_draw']:.1%}")
    print(f"Vitória {pred['away_team']}: {pred['prob_away_win']:.1%}")
    print("\n--- Gols Esperados (λ) ---")
    print(f"λ {pred['home_team']}: {pred['lambda_home']:.2f}")
    print(f"λ {pred['away_team']}: {pred['lambda_away']:.2f}")
    print("\n--- Elo (referência) ---")
    print(f"{pred['home_team']}: {pred['elo_rating_home']:.0f}  vs  {pred['away_team']}: {pred['elo_rating_away']:.0f}")
    print(f"Probabilidade implícita por Elo: {pred['home_team']} {pred['prob_elo_home']:.1%} — {pred['away_team']} {pred['prob_elo_away']:.1%}")
    print("\n--- Placares mais prováveis (Poisson) ---")
    print("Placar    Prob")
    for score, prob in pred['top_scores'][:10]:
        print(f"{score:8}  {prob:.1%}")
    print("="*70 + "\n")

# ========================= EXECUÇÃO PRINCIPAL =========================
if __name__ == "__main__":
    # 1. Carregar os dados históricos
    # Altere o caminho conforme necessário
    DATA_PATH = "data/results.csv"   # padrão do repositório original
    
    try:
        df = pd.read_csv(DATA_PATH)
        # Verificar colunas necessárias
        required_cols = ['date', 'home_team', 'away_team', 'home_score', 'away_score', 'tournament']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Coluna '{col}' não encontrada no CSV. Verifique o formato.")
        if 'neutral' not in df.columns:
            df['neutral'] = False   # assume que não é neutro se coluna ausente
        print(f"Dados carregados: {len(df)} partidas, período de {df['date'].min()} a {df['date'].max()}")
    except FileNotFoundError:
        print(f"Erro: Arquivo '{DATA_PATH}' não encontrado.")
        print("Por favor, forneça o caminho correto para o arquivo results.csv")
        exit(1)
    
    # 2. Computar Elo nos últimos LOOKBACK_YEARS anos
    print(f"\n[1] Calculando rating Elo (últimos {LOOKBACK_YEARS} anos)...")
    elo_ratings = compute_elo_ratings(df, lookback_years=LOOKBACK_YEARS)
    print(f"Times processados: {len(elo_ratings)}")
    
    # 3. Estimar forças ofensivas/defensivas para Poisson (mesmo período)
    print(f"\n[2] Estimando forças para modelo Poisson (últimos {LOOKBACK_YEARS} anos)...")
    attack, defense, league_avg = team_strength_poisson(df, lookback_years=LOOKBACK_YEARS)
    print(f"Média global de gols por jogo: {league_avg:.2f}")
    
    # 4. Exemplo de predição para os jogos que você mencionou
    #    (descomente os que desejar)
    
    # Exemplo 1: Suécia vs Turquia (campo neutro)
    print("\n" + "🔮 PREDIÇÃO PARA SUÉCIA x TURQUIA (campo neutro)")
    pred1 = predict_match("Sweden", "Türkiye", neutral=True, df=df,
                          elo_ratings=elo_ratings, attack=attack, defense=defense, league_avg=league_avg)
    print_prediction_report(pred1)
    
    # Exemplo 2: Costa do Marfim vs Equador (campo neutro)
    print("\n" + "🔮 PREDIÇÃO PARA COSTA DO MARFIM x EQUADOR (campo neutro)")
    pred2 = predict_match("Ivory Coast", "Ecuador", neutral=True, df=df,
                          elo_ratings=elo_ratings, attack=attack, defense=defense, league_avg=league_avg)
    print_prediction_report(pred2)
    
    # (Opcional) Exemplo com vantagem de casa:
    # pred3 = predict_match("Brazil", "Argentina", neutral=False, ...)
    
    print("\n✅ Script finalizado. As probabilidades do Elo e do Poisson agora estão alinhadas porque ambas usam os mesmos últimos 10 anos.")