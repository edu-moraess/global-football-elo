"""
Club & Player Market Intelligence Engine
Dataset: Transfermarkt (Kaggle) — players, clubs, transfers, player_valuations, competitions
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data_clubs"

COMPETITION_LABELS = {
    "GB1": "Premier League", "ES1": "La Liga", "IT1": "Serie A",
    "L1": "Bundesliga", "FR1": "Ligue 1", "PO1": "Liga Portugal",
    "NL1": "Eredivisie", "TR1": "Süper Lig", "BRA1": "Brasileirão",
    "RU1": "Premier Liga (RUS)", "GR1": "Super League (GRE)",
    "UKR1": "Premier Liga (UKR)", "ARG1": "Liga Profesional (ARG)",
}


def load_club_data():
    players = pd.read_csv(DATA_DIR / "players.csv", low_memory=False)
    clubs = pd.read_csv(DATA_DIR / "clubs.csv", low_memory=False)
    transfers = pd.read_csv(DATA_DIR / "transfers.csv", low_memory=False)
    valuations = pd.read_csv(DATA_DIR / "player_valuations.csv", low_memory=False)
    competitions = pd.read_csv(DATA_DIR / "competitions.csv", low_memory=False)

    players["date_of_birth"] = pd.to_datetime(players["date_of_birth"], errors="coerce")
    transfers["transfer_date"] = pd.to_datetime(transfers["transfer_date"], errors="coerce")
    valuations["date"] = pd.to_datetime(valuations["date"], errors="coerce")
    transfers["transfer_fee"] = pd.to_numeric(transfers["transfer_fee"], errors="coerce").fillna(0)
    transfers["market_value_in_eur"] = pd.to_numeric(transfers["market_value_in_eur"], errors="coerce")

    # league display names
    competitions["league_name"] = competitions["competition_id"].map(COMPETITION_LABELS).fillna(competitions["name"])

    return {
        "players": players,
        "clubs": clubs,
        "transfers": transfers,
        "valuations": valuations,
        "competitions": competitions,
    }


def player_age_at(players: pd.DataFrame, ref_date: pd.Timestamp):
    return (ref_date - players["date_of_birth"]).dt.days / 365.25


def valuation_age_curve(valuations: pd.DataFrame, players: pd.DataFrame, position_filter=None, min_value=10000):
    """Build market-value-by-age curve, optionally filtered by position."""
    pl = players[["player_id", "date_of_birth", "position", "sub_position"]].copy()
    v = valuations.merge(pl, on="player_id", how="inner")
    v = v.dropna(subset=["date_of_birth"])
    v["age"] = (v["date"] - v["date_of_birth"]).dt.days / 365.25
    v = v[(v["age"] >= 15) & (v["age"] <= 42) & (v["market_value_in_eur"] >= min_value)]

    if position_filter and position_filter != "Todas":
        v = v[v["position"] == position_filter]

    v["age_bucket"] = v["age"].round().astype(int)
    curve = v.groupby("age_bucket")["market_value_in_eur"].agg(["mean", "median", "count"]).reset_index()
    curve = curve[curve["count"] >= 10]
    return curve


def top_transfers(transfers: pd.DataFrame, n=20, min_fee=1_000_000):
    t = transfers[transfers["transfer_fee"] >= min_fee].copy()
    t = t.sort_values("transfer_fee", ascending=False).head(n)
    return t[["transfer_date", "player_name", "from_club_name", "to_club_name", "transfer_fee", "transfer_season"]]


def transfer_flow_by_league(transfers: pd.DataFrame, clubs: pd.DataFrame, season=None, top_n=12, min_fee=500_000):
    """Build Sankey-ready flow of transfer fees between domestic competitions (leagues)."""
    club_to_comp = clubs.set_index("club_id")["domestic_competition_id"].to_dict()

    t = transfers.copy()
    if season:
        t = t[t["transfer_season"] == season]
    t = t[t["transfer_fee"] >= min_fee]

    t["from_league"] = t["from_club_id"].map(club_to_comp)
    t["to_league"] = t["to_club_id"].map(club_to_comp)
    t = t.dropna(subset=["from_league", "to_league"])
    t = t[t["from_league"] != t["to_league"]]

    flow = t.groupby(["from_league", "to_league"])["transfer_fee"].sum().reset_index()
    flow = flow.sort_values("transfer_fee", ascending=False).head(top_n)
    flow["from_league"] = flow["from_league"].map(COMPETITION_LABELS).fillna(flow["from_league"])
    flow["to_league"] = flow["to_league"].map(COMPETITION_LABELS).fillna(flow["to_league"])
    return flow


def nationality_distribution(players: pd.DataFrame, competition_id=None):
    p = players.copy()
    if competition_id and competition_id != "Todas":
        p = p[p["current_club_domestic_competition_id"] == competition_id]
    dist = p["country_of_citizenship"].value_counts().reset_index()
    dist.columns = ["País", "Jogadores"]
    return dist


def club_summary(clubs: pd.DataFrame, players: pd.DataFrame, competition_id=None):
    c = clubs.copy()
    if competition_id and competition_id != "Todas":
        c = c[c["domestic_competition_id"] == competition_id]

    val_by_club = players.groupby("current_club_id")["market_value_in_eur"].sum().to_dict()
    c["squad_value_eur"] = c["club_id"].map(val_by_club)
    return c[["name", "squad_size", "average_age", "foreigners_percentage",
              "national_team_players", "squad_value_eur", "stadium_name", "stadium_seats"]].dropna(subset=["squad_value_eur"])
