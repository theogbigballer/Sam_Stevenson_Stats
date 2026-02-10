import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import altair as alt
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

BASKETBALL_SENIOR_URL = (
    "https://sports.deseret.com/high-school/boys-basketball/athlete/sam-stevenson/301453"
)
BASKETBALL_UNDERCLASS_URL = (
    "https://sports.deseret.com/high-school/boys-basketball/athlete/stevenson/233594"
)
FOOTBALL_URL = "https://sports.deseret.com/high-school/football/athlete/stevenson/233594"
DAVID_BASKETBALL_URL = (
    "https://sports.deseret.com/high-school/boys-basketball/athlete/david-stevenson/208619"
)

SEASON_PATTERN = re.compile(r"(\d{4})\s*-\s*(\d{4})")
SEASON_SHORT_PATTERN = re.compile(r"(\d{4})\s*-\s*(\d{2})")


@st.cache_data(ttl=3600)
def fetch_html(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36"
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.text


def _flatten_column(col) -> str:
    if isinstance(col, tuple):
        parts = [str(part) for part in col if str(part).strip() and str(part).lower() != "nan"]
        upper_parts = [part.strip().upper() for part in parts]
        if "SEASON" in upper_parts:
            return "SEASON"
        if parts:
            return parts[-1].strip()
        return ""
    text = str(col).strip()
    if text.startswith("(") and text.endswith(")") and "," in text:
        # Handle stringified tuple like "('Season Stats', 'SEASON', 'GAME')"
        parts = [part.strip().strip("'\"") for part in text[1:-1].split(",")]
        parts = [part for part in parts if part and part.lower() != "nan"]
        upper_parts = [part.strip().upper() for part in parts]
        if "SEASON" in upper_parts:
            return "SEASON"
        if parts:
            return parts[-1]
    return text


def _clean_columns(columns: List[str]) -> List[str]:
    return [_flatten_column(col) for col in columns]


def _normalize_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _extract_season(text: str) -> str:
    match = SEASON_PATTERN.search(text)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    match = SEASON_SHORT_PATTERN.search(text)
    if match:
        start = int(match.group(1))
        end_two = int(match.group(2))
        end = (start // 100) * 100 + end_two
        if end < start:
            end += 100
        return f"{start}-{end}"
    return text.strip()


def _find_table_by_schema(
    tables: List[pd.DataFrame], schema: Dict[str, List[str]]
) -> Optional[pd.DataFrame]:
    for table in tables:
        columns = _clean_columns(table.columns)
        norm_map: Dict[str, str] = {}
        for col in columns:
            norm_map[_normalize_col(col)] = col

        rename_map: Dict[str, str] = {}
        for target, options in schema.items():
            found = None
            for option in options:
                option_norm = _normalize_col(option)
                if option_norm in norm_map:
                    found = norm_map[option_norm]
                    break
                for norm, original in norm_map.items():
                    if option_norm and option_norm in norm:
                        found = original
                        break
                if found:
                    break
            if not found:
                break
            rename_map[found] = target
        else:
            matched = table.copy()
            matched.columns = columns
            matched.rename(columns=rename_map, inplace=True)
            return matched
    return None


def _read_tables(html: str) -> List[pd.DataFrame]:
    try:
        return pd.read_html(html)
    except ValueError:
        return []


@dataclass
class SeasonTotals:
    season: str
    games: int
    points: int
    threes: int
    rebounds: int
    assists: int
    steals: int
    blocks: int


@dataclass
class GameLog:
    season: str
    games: pd.DataFrame


def parse_basketball_season_totals(html: str) -> List[SeasonTotals]:
    tables = _read_tables(html)
    season_table = _find_table_by_schema(
        tables,
        {
            "SEASON": ["SEASON"],
            "POINTS": ["POINTS", "PTS"],
            "3PT": ["3PT", "3P", "3PTFG", "3PTM"],
            "REBOUNDS": ["REBOUNDS", "REB"],
            "ASSISTS": ["ASSISTS", "AST"],
            "STEALS": ["STEALS", "STL"],
            "BLOCKS": ["BLOCKS", "BLK"],
        },
    )
    if season_table is None:
        # Fallback: find table by flattened column scan
        for table in tables:
            cols = _clean_columns(table.columns)
            if "SEASON" in cols and "POINTS" in cols:
                season_table = table.copy()
                season_table.columns = cols
                break
    if season_table is None:
        return []

    season_table = season_table.copy()
    if "SEASON" not in season_table.columns:
        return []
    season_table["SEASON"] = season_table["SEASON"].astype(str)
    season_table = season_table[season_table["SEASON"].apply(_is_season_header)]
    season_table["SEASON"] = season_table["SEASON"].apply(_extract_season)
    if "GAME" not in season_table.columns:
        season_table["GAME"] = pd.NA

    def _safe_int(value) -> int:
        num = pd.to_numeric(value, errors="coerce")
        if pd.isna(num):
            return 0
        return int(num)

    totals = []
    for _, row in season_table.iterrows():
        totals.append(
            SeasonTotals(
                season=row["SEASON"],
                games=_safe_int(row.get("GAME")),
                points=_safe_int(row.get("POINTS")),
                threes=_safe_int(row.get("3PT")),
                rebounds=_safe_int(row.get("REBOUNDS")),
                assists=_safe_int(row.get("ASSISTS")),
                steals=_safe_int(row.get("STEALS")),
                blocks=_safe_int(row.get("BLOCKS")),
            )
        )
    return totals


def parse_basketball_game_log(html: str, season: str) -> Optional[GameLog]:
    tables = _read_tables(html)
    game_table = _find_table_by_schema(
        tables,
        {
            "GAME": ["GAME", "DATE"],
            "POINTS": ["POINTS", "PTS"],
            "3PT": ["3PT", "3P", "3PTFG", "3PTM"],
            "REBOUNDS": ["REBOUNDS", "REB"],
            "ASSISTS": ["ASSISTS", "AST"],
            "STEALS": ["STEALS", "STL"],
            "BLOCKS": ["BLOCKS", "BLK"],
        },
    )
    if game_table is None:
        # Fallback: look for a table with GAME + POINTS and without SEASON
        for table in tables:
            cols = _clean_columns(table.columns)
            if "GAME" in cols and "POINTS" in cols and "SEASON" not in cols:
                game_table = table.copy()
                game_table.columns = cols
                break
    if game_table is None:
        return None

    df = game_table.copy()
    df.rename(
        columns={
            "GAME": "Game",
            "POINTS": "Points",
            "3PT": "Threes",
            "REBOUNDS": "Rebounds",
            "ASSISTS": "Assists",
            "STEALS": "Steals",
            "BLOCKS": "Blocks",
        },
        inplace=True,
    )
    if "Game" not in df.columns:
        return None

    # Remove header/summary rows so only actual games remain
    df["Game"] = df["Game"].astype(str)
    df = df[~df["Game"].str.contains("GAME", case=False, na=False)]
    df = df[~df["Game"].str.contains("SEASON", case=False, na=False)]
    df = df[df["Game"].str.contains(r"\d{1,2}/\d{1,2}", regex=True, na=False)]

    df["Date"] = df["Game"].astype(str).str.extract(r"^(\d{1,2}/\d{1,2})")
    df["Opponent"] = df["Game"].astype(str).str.replace(
        r"^\d{1,2}/\d{1,2}", "", regex=True
    )
    df["Opponent"] = df["Opponent"].str.replace("@", "@ ", regex=False).str.strip()
    return GameLog(season=season, games=df)


def _finalize_game_log(df: pd.DataFrame, season: str) -> GameLog:
    df = df.copy()
    df["Date"] = df["Game"].astype(str).str.extract(r"^(\d{1,2}/\d{1,2})")
    df["Opponent"] = df["Game"].astype(str).str.replace(
        r"^\d{1,2}/\d{1,2}", "", regex=True
    )
    df["Opponent"] = df["Opponent"].str.replace("@", "@ ", regex=False).str.strip()
    return GameLog(season=season, games=df)


def _is_season_header(text: str) -> bool:
    return bool(SEASON_PATTERN.search(text) or SEASON_SHORT_PATTERN.search(text))


def parse_basketball_game_logs_multi(html: str, season_hint: Optional[str] = None) -> List[GameLog]:
    tables = _read_tables(html)
    game_table = _find_table_by_schema(
        tables,
        {
            "GAME": ["GAME", "DATE"],
            "POINTS": ["POINTS", "PTS"],
            "3PT": ["3PT", "3P", "3PTFG", "3PTM"],
            "REBOUNDS": ["REBOUNDS", "REB"],
            "ASSISTS": ["ASSISTS", "AST"],
            "STEALS": ["STEALS", "STL"],
            "BLOCKS": ["BLOCKS", "BLK"],
        },
    )
    if game_table is None:
        for table in tables:
            cols = _clean_columns(table.columns)
            if "GAME" in cols and "POINTS" in cols and "SEASON" not in cols:
                game_table = table.copy()
                game_table.columns = cols
                break
    if game_table is None:
        return []

    df = game_table.copy()
    df.rename(
        columns={
            "GAME": "Game",
            "POINTS": "Points",
            "3PT": "Threes",
            "REBOUNDS": "Rebounds",
            "ASSISTS": "Assists",
            "STEALS": "Steals",
            "BLOCKS": "Blocks",
        },
        inplace=True,
    )
    if "Game" not in df.columns:
        return []

    logs: Dict[str, List[Dict[str, object]]] = {}
    current_season = season_hint
    for _, row in df.iterrows():
        game_cell = str(row["Game"])
        if _is_season_header(game_cell):
            current_season = _extract_season(game_cell)
            continue
        if not re.search(r"\d{1,2}/\d{1,2}", game_cell):
            continue
        if not current_season:
            current_season = season_hint or "Unknown"

        logs.setdefault(current_season, []).append(
            {
                "Game": game_cell,
                "Points": row.get("Points"),
                "Threes": row.get("Threes"),
                "Rebounds": row.get("Rebounds"),
                "Assists": row.get("Assists"),
                "Steals": row.get("Steals"),
                "Blocks": row.get("Blocks"),
            }
        )

    results: List[GameLog] = []
    for season, rows in logs.items():
        season_df = pd.DataFrame(rows)
        season_df["Date"] = season_df["Game"].astype(str).str.extract(r"^(\d{1,2}/\d{1,2})")
        season_df["Opponent"] = season_df["Game"].astype(str).str.replace(
            r"^\d{1,2}/\d{1,2}", "", regex=True
        )
        season_df["Opponent"] = season_df["Opponent"].str.replace("@", "@ ", regex=False).str.strip()
        results.append(GameLog(season=season, games=season_df))

    return results


def build_basketball_data() -> Tuple[List[SeasonTotals], List[GameLog]]:
    senior_html = fetch_html(BASKETBALL_SENIOR_URL)
    underclass_html = fetch_html(BASKETBALL_UNDERCLASS_URL)

    totals = []
    totals.extend(parse_basketball_season_totals(senior_html))
    totals.extend(parse_basketball_season_totals(underclass_html))

    logs: List[GameLog] = []
    logs.extend(parse_basketball_game_logs_multi(senior_html, season_hint="2022-2023"))
    logs.extend(parse_basketball_game_logs_multi(underclass_html, season_hint="2021-2022"))

    totals = {t.season: t for t in totals}
    sorted_totals = [totals[key] for key in sorted(totals.keys())]

    return sorted_totals, logs


def build_david_basketball_data() -> Tuple[List[SeasonTotals], List[GameLog]]:
    html = fetch_html(DAVID_BASKETBALL_URL)
    totals = parse_basketball_season_totals(html)
    logs = parse_basketball_game_logs_multi(html)
    totals = {t.season: t for t in totals}
    sorted_totals = [totals[key] for key in sorted(totals.keys())]

    # If logs exist but seasons are unknown, assume they belong to latest season listed.
    if logs and all(log.season in (None, "", "Unknown") for log in logs) and sorted_totals:
        def _season_key(season: str) -> int:
            match = re.match(r"(\\d{4})", season or "")
            return int(match.group(1)) if match else 0

        latest_season = sorted(sorted_totals, key=lambda t: _season_key(t.season))[-1].season
        for log in logs:
            log.season = latest_season

    # Inject manual 2017-2018 junior game log (provided by user).
    if not any(log.season == "2017-2018" for log in logs):
        manual_rows = [
            ("11/21@Davis", 0, 0, 1, 2, 2, 0),
            ("11/30Lehi", 3, 1, 3, 4, 1, 0),
            ("12/1Pleasant Grove", 6, 0, 2, 1, 1, 0),
            ("12/2@Corner Canyon", 4, 0, 1, 4, 2, 0),
            ("12/7Copper Hills", 6, 0, 4, 1, 0, 0),
            ("12/8@Brighton", 8, 0, 1, 4, 0, 0),
            ("12/9@Bingham", 2, 0, 1, 3, 0, 0),
            ("12/12Lehi", 4, 0, 3, 2, 0, 0),
            ("12/20Villa Park, Calif.", 8, 0, 3, 5, 4, 0),
            ("12/21Dominguez, Calif.", 3, 1, 1, 5, 0, 0),
            ("12/22Timpanogos", 5, 1, 1, 3, 1, 0),
            ("12/23Valor Christian, Colo.", 2, 0, 2, 2, 0, 0),
            ("1/5Springville", 7, 2, 2, 5, 1, 0),
            ("1/9@Olympus", 6, 0, 5, 0, 2, 0),
            ("1/23Roy", 7, 1, 1, 2, 3, 0),
            ("1/26@Woods Cross", 3, 1, 0, 3, 0, 0),
            ("1/30Box Elder", 3, 0, 1, 3, 1, 0),
            ("2/2Viewmont", 2, 0, 2, 2, 0, 0),
            ("2/6@Roy", 11, 0, 1, 4, 1, 0),
            ("2/9@Box Elder", 2, 0, 0, 2, 1, 0),
            ("2/16Woods Cross", 0, 0, 1, 1, 1, 0),
            ("2/20@Viewmont", 0, 0, 1, 1, 0, 0),
            ("2/26@Skyridge", 8, 1, 0, 0, 1, 0),
            ("2/28Skyline", 10, 0, 3, 2, 0, 0),
            ("3/2@Olympus", 2, 0, 0, 2, 0, 0),
        ]
        manual_df = pd.DataFrame(
            manual_rows,
            columns=["Game", "Points", "Threes", "Rebounds", "Assists", "Steals", "Blocks"],
        )
        logs.append(_finalize_game_log(manual_df, "2017-2018"))
    return sorted_totals, logs


def compute_per_game_from_log(log: GameLog) -> Dict[str, float]:
    numeric_cols = ["Points", "Threes", "Rebounds", "Assists", "Steals", "Blocks"]
    df = log.games.copy()
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=numeric_cols, how="all")
    games = len(df)
    if games == 0:
        return {}
    return {
        "Season": log.season,
        "Games": games,
        "PPG": df["Points"].mean(),
        "3PG": df["Threes"].mean(),
        "RPG": df["Rebounds"].mean(),
        "APG": df["Assists"].mean(),
        "SPG": df["Steals"].mean(),
        "BPG": df["Blocks"].mean(),
    }


def compute_per_game_from_totals(total: SeasonTotals) -> Dict[str, float]:
    games = 23
    return {
        "Season": total.season,
        "Games": games,
        "PPG": total.points / games,
        "3PG": total.threes / games,
        "RPG": total.rebounds / games,
        "APG": total.assists / games,
        "SPG": total.steals / games,
        "BPG": total.blocks / games,
    }


def _career_highs(logs: List[GameLog]) -> Dict[str, Dict[str, object]]:
    rows = []
    for log in logs:
        if log.games is None or log.games.empty:
            continue
        df = log.games.copy()
        for col in ["Points", "Rebounds", "Assists", "Steals", "Blocks", "Threes"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["Season"] = log.season
        rows.append(df)
    if not rows:
        return {}

    all_games = pd.concat(rows, ignore_index=True)
    highs = {}
    for col, label in [
        ("Points", "PPG"),
        ("Rebounds", "RPG"),
        ("Assists", "APG"),
        ("Steals", "SPG"),
        ("Blocks", "BPG"),
        ("Threes", "3PG"),
    ]:
        if col not in all_games.columns:
            continue
        candidate = all_games.dropna(subset=[col])
        if candidate.empty:
            continue
        best = candidate.sort_values(col, ascending=False).iloc[0]
        highs[label] = {
            "value": int(best[col]),
            "date": best.get("Date"),
            "season": best.get("Season"),
            "opponent": best.get("Opponent"),
        }
    return highs


def parse_football_receiver_stats(html: str) -> Dict[str, Dict[str, str]]:
    tables = _read_tables(html)
    receiving = _find_table_by_schema(
        tables,
        {
            "SEASON": ["SEASON"],
            "RECEPTIONS": ["RECEPTIONS", "REC"],
            "YARDS": ["YARDS", "YRDS", "YDS"],
            "YARDS/RECEP.": ["YARDS/RECEP.", "YARDS/REC", "YPR"],
            "TD": ["TD"],
        },
    )
    if receiving is None:
        # Fallback: match receiving table even if no SEASON column exists
        for table in tables:
            cols = _clean_columns(table.columns)
            upper_cols = {c.upper() for c in cols}
            if {"RECEPTIONS", "YARDS", "TD"}.issubset(upper_cols):
                receiving = table.copy()
                receiving.columns = cols
                break

    results: Dict[str, Dict[str, str]] = {}
    if receiving is not None:
        receiving = receiving.copy()
        if "SEASON" in receiving.columns:
            receiving["SEASON"] = receiving["SEASON"].astype(str).apply(_extract_season)
            for _, row in receiving.iterrows():
                season_label = str(row["SEASON"])
                if season_label.startswith("2022"):
                    results["Receiving (2022-2023)"] = {
                        col.title(): str(row[col]) for col in receiving.columns
                    }
            if not results and not receiving.empty:
                row = receiving.iloc[0]
                results["Receiving (latest)"] = {
                    col.title(): str(row[col]) for col in receiving.columns
                }
        elif not receiving.empty:
            row = receiving.iloc[0]
            results["Receiving (latest)"] = {
                col.title(): str(row[col]) for col in receiving.columns
            }
    return results


def parse_football_game_log(html: str) -> Optional[pd.DataFrame]:
    tables = _read_tables(html)
    game_table = _find_table_by_schema(
        tables,
        {
            "GAME": ["GAME", "DATE"],
            "RECEPTIONS": ["RECEPTIONS", "REC"],
            "YARDS": ["YARDS", "YRDS", "YDS"],
            "TD": ["TD"],
        },
    )
    if game_table is None:
        for table in tables:
            cols = _clean_columns(table.columns)
            upper_cols = {c.upper() for c in cols}
            if "GAME" in upper_cols and "RECEPTIONS" in upper_cols:
                game_table = table.copy()
                game_table.columns = cols
                break
    if game_table is None:
        return None

    df = game_table.copy()
    df.rename(
        columns={
            "GAME": "Game",
            "RECEPTIONS": "Receptions",
            "YARDS": "Yards",
            "TD": "TD",
        },
        inplace=True,
    )
    if "Game" not in df.columns:
        return None

    df["Game"] = df["Game"].astype(str)
    df = df[~df["Game"].str.contains("GAME", case=False, na=False)]
    df = df[df["Game"].str.contains(r"\d{1,2}/\d{1,2}", regex=True, na=False)]
    df["Date"] = df["Game"].astype(str).str.extract(r"^(\d{1,2}/\d{1,2})")
    df["Opponent"] = df["Game"].astype(str).str.replace(
        r"^\d{1,2}/\d{1,2}", "", regex=True
    )
    df["Opponent"] = df["Opponent"].str.replace("@", "@ ", regex=False).str.strip()
    return df


def _best_games(logs: List[GameLog], top_n: int = 3) -> List[str]:
    rows = []
    for log in logs:
        if log.games is None or log.games.empty:
            continue
        df = log.games.copy()
        for col in ["Points", "Rebounds", "Assists", "Steals", "Blocks", "Threes"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["Season"] = log.season
        rows.append(df)
    if not rows:
        return []

    all_games = pd.concat(rows, ignore_index=True)
    highlights = []

    def pick_best(stat: str, label: str) -> Optional[str]:
        if stat not in all_games.columns:
            return None
        candidate = all_games.dropna(subset=[stat])
        if candidate.empty:
            return None
        best = candidate.sort_values(stat, ascending=False).iloc[0]
        result_suffix = ""
        result_value = None
        for col in ["Result", "RESULT"]:
            if col in best and pd.notna(best[col]):
                result_value = str(best[col]).strip()
                break
        if result_value and result_value.upper().startswith("W"):
            result_suffix = f" ({result_value})"
        return (
            f"{label}: {int(best[stat])} vs {best['Opponent']} "
            f"({best['Date']}, {best['Season']}){result_suffix}"
        )

    for stat, label in [
        ("Points", "Top scoring game"),
        ("Assists", "Top assists game"),
        ("Rebounds", "Top rebounds game"),
        ("Steals", "Top steals game"),
        ("Blocks", "Top blocks game"),
        ("Threes", "Top 3PT game"),
    ]:
        line = pick_best(stat, label)
        if line:
            highlights.append(line)

    return highlights[:top_n] if top_n > 0 else highlights


def _normalize_opponent(name: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(name)).strip()
    cleaned = cleaned.replace("@ ", "").replace("@", "").strip()
    return cleaned


def build_opponent_games(logs: List[GameLog]) -> pd.DataFrame:
    rows = []
    for log in logs:
        if log.games is None or log.games.empty:
            continue
        df = log.games.copy()
        df["HomeAway"] = df["Opponent"].apply(
            lambda name: "Away" if str(name).strip().startswith("@") else "Home"
        )
        for col in ["Points", "Rebounds", "Assists", "Steals", "Blocks", "Threes"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["Season"] = log.season
        df["Opponent"] = df["Opponent"].apply(_normalize_opponent)
        rows.append(df)
    if not rows:
        return pd.DataFrame()

    all_games = pd.concat(rows, ignore_index=True)
    all_games.rename(
        columns={
            "Points": "PPG",
            "Threes": "3PG",
            "Rebounds": "RPG",
            "Assists": "APG",
            "Steals": "SPG",
            "Blocks": "BPG",
        },
        inplace=True,
    )
    return all_games


def _summarize_opponents(opponent_games: pd.DataFrame) -> List[str]:
    if opponent_games.empty:
        return []

    summaries = []
    for opponent, group in opponent_games.groupby("Opponent"):
        group = group.copy()
        for stat in ["PPG", "RPG", "APG", "SPG", "BPG", "3PG"]:
            group[stat] = pd.to_numeric(group[stat], errors="coerce")

        games_played = len(group)
        if games_played == 0:
            continue

        avg_ppg = group["PPG"].mean()
        avg_rpg = group["RPG"].mean()
        avg_apg = group["APG"].mean()
        avg_spg = group["SPG"].mean()
        avg_bpg = group["BPG"].mean()
        avg_3pg = group["3PG"].mean()

        best_pts = group.sort_values("PPG", ascending=False).iloc[0]
        best_pts_line = (
            f"best scoring game {int(best_pts['PPG'])} "
            f"({best_pts['Date']}, {best_pts['Season']})"
        )

        home_away_counts = group["HomeAway"].value_counts()
        home = int(home_away_counts.get("Home", 0))
        away = int(home_away_counts.get("Away", 0))

        summaries.append(
            f"Vs {opponent}: {games_played} games, {avg_ppg:.1f} PPG, {avg_rpg:.1f} RPG, "
            f"{avg_apg:.1f} APG, {avg_spg:.1f} SPG, {avg_bpg:.1f} BPG, {avg_3pg:.1f} 3PG; "
            f"{best_pts_line}. Home/Away split: {home} home, {away} away."
        )

    return summaries


def parse_profile(html: str) -> Dict[str, str]:
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(" ", strip=True)

    def extract_field(label: str) -> str:
        pattern = rf"{label}\s*([A-Za-z0-9' ]{{1,20}})"
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        return ""

    return {
        "Class": extract_field("CLASS"),
        "Height": extract_field("HEIGHT"),
        "School": extract_field("SCHOOL"),
        "Position": extract_field("POSITION"),
    }


def render_per_game_chart(per_game_df: pd.DataFrame, key: str) -> None:
    stat_order = ["PPG", "3PG", "RPG", "APG", "SPG", "BPG"]
    chart_data = per_game_df.melt(
        id_vars=["Season"],
        value_vars=stat_order,
        var_name="Stat",
        value_name="Value",
    )

    stat_selection = st.multiselect(
        "Select stats for the year-by-year graph",
        stat_order,
        default=["PPG", "RPG", "APG"],
        key=f"stat_select_{key}",
    )
    if not stat_selection:
        st.info("Select at least one stat to render the graph.")
        return

    filtered = chart_data[chart_data["Stat"].isin(stat_selection)]
    chart = (
        alt.Chart(filtered)
        .mark_line(point=True)
        .encode(
            x=alt.X("Season:N", sort=sorted(per_game_df["Season"].unique()), title=None),
            y=alt.Y("Value:Q", title="Per game"),
            color=alt.Color("Stat:N", sort=stat_order, title="Stat"),
            tooltip=["Season:N", "Stat:N", alt.Tooltip("Value:Q", format=".2f")],
        )
    )
    st.altair_chart(chart, use_container_width=True)


def improvement_summary(per_game_df: pd.DataFrame) -> List[str]:
    if len(per_game_df) < 2:
        return ["Not enough per-game data to calculate year-over-year change."]

    per_game_df = per_game_df.sort_values("Season")
    prev = per_game_df.iloc[-2]
    curr = per_game_df.iloc[-1]

    summaries = []
    for stat in ["PPG", "3PG", "RPG", "APG", "SPG", "BPG"]:
        if pd.isna(prev[stat]) or pd.isna(curr[stat]):
            continue
        delta = curr[stat] - prev[stat]
        if prev[stat] == 0:
            pct = "n/a"
        else:
            pct = f"{(delta / prev[stat]) * 100:.1f}%"
        direction = "up" if delta > 0 else "down" if delta < 0 else "flat"
        summaries.append(f"{stat}: {direction} {delta:+.2f} ({pct})")

    return summaries


def main() -> None:
    st.set_page_config(page_title="Sam Stevenson Player Analysis", layout="wide")

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Inter:wght@400;600&display=swap');
        :root { --bg: #0a0a0c; --panel: #111219; --panel-2: #151826; --text: #f4f5f7; --muted: #b7bac4; --accent: #ff4d4d; }
        .stApp { background: radial-gradient(circle at 20% -10%, #1a1c2b 0%, #0a0a0c 45%); color: var(--text); font-family: "Space Grotesk", system-ui, sans-serif; }
        [data-testid="stHeader"] { background: rgba(0,0,0,0); }
        [data-testid="stSidebar"] { background-color: #0f111a; }
        h1, h2, h3, h4 { color: var(--text); font-family: "Space Grotesk", system-ui, sans-serif; }
        p, span, li, label, .stMarkdown, .stText, .stCaption { color: var(--muted); }
        .stCaption { color: #9aa0af; }
        .stTitle { font-weight: 700; letter-spacing: 0.3px; }
        .stTabs [role="tab"] { color: var(--muted); }
        .stTabs [role="tab"][aria-selected="true"] { color: var(--text); border-bottom: 2px solid var(--accent); }
        .stButton > button { background: var(--accent); color: #0b0b0b; border: none; border-radius: 10px; font-weight: 700; }
        .stButton > button:hover { background: #ff6b6b; color: #0b0b0b; }
        [data-testid="stMetric"] { background: var(--panel); border-radius: 12px; padding: 12px; border: 1px solid #1f2333; }
        [data-testid="stMetric"] * { color: var(--text); }
        .stDataFrame, .stTable { background: var(--panel-2); border-radius: 12px; border: 1px solid #1f2333; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Sam Stevenson Player Analysis")
    st.caption("Data sources: Deseret News high school sports profiles (basketball and football).")

    if st.button("Refresh data"):
        st.cache_data.clear()

    show_diagnostics = st.checkbox("Show parsing diagnostics", value=False)

    try:
        totals, logs = build_basketball_data()
        david_totals, david_logs = build_david_basketball_data()
    except Exception as exc:
        st.error(f"Unable to load basketball data: {exc}")
        return

    if show_diagnostics:
        st.subheader("Diagnostics: Basketball Tables")
        try:
            senior_html = fetch_html(BASKETBALL_SENIOR_URL)
            underclass_html = fetch_html(BASKETBALL_UNDERCLASS_URL)
            david_html = fetch_html(DAVID_BASKETBALL_URL)
            for label, html in [
                ("Senior Page", senior_html),
                ("Underclass Page", underclass_html),
                ("David Page", david_html),
            ]:
                tables = _read_tables(html)
                with st.expander(f"{label} ({len(tables)} tables)"):
                    for idx, table in enumerate(tables):
                        cols = [str(col) for col in table.columns]
                        st.write(f"Table {idx + 1} columns: {cols}")
                totals_preview = parse_basketball_season_totals(html)
                if totals_preview:
                    seasons = [item.season for item in totals_preview]
                    st.caption(f"{label} seasons parsed: {seasons}")
                else:
                    st.caption(f"{label} seasons parsed: none")
        except Exception as exc:
            st.warning(f"Diagnostics failed: {exc}")

    tabs = st.tabs(
        ["Season Overview", "Opponent Comparison", "David Overview", "David Opponents", "Sibling Comparison"]
    )

    with tabs[0]:
        st.subheader("Basketball Overview")

        profile = {}
        if totals:
            try:
                profile = parse_profile(fetch_html(BASKETBALL_SENIOR_URL))
            except Exception:
                profile = {}

        if profile:
            cols = st.columns(4)
            for col, label in zip(cols, ["Class", "Height", "School", "Position"]):
                value = profile.get(label, "") or "—"
                col.metric(label, value)

        if totals:
            totals_df = pd.DataFrame([t.__dict__ for t in totals])
            st.write("Season totals")
            st.dataframe(totals_df, use_container_width=True)

        per_game_rows = [compute_per_game_from_totals(total) for total in totals]
        per_game_rows = [row for row in per_game_rows if row]

        if per_game_rows:
            per_game_df = pd.DataFrame(per_game_rows)
            st.write("Per-game averages (based on 23-game seasons)")
            st.dataframe(per_game_df, use_container_width=True)
            render_per_game_chart(per_game_df, key="sam_overview")

            st.subheader("Improvement Highlights")
            for line in improvement_summary(per_game_df):
                st.write(f"- {line}")
        else:
            st.info("Season totals not found, so per-game averages are unavailable.")

        st.subheader("Basketball Game Logs By Season")
        season_order = ["2022-2023", "2021-2022", "2020-2021", "2019-2020"]
        season_labels = {
            "2022-2023": "Senior Year (2022-2023)",
            "2021-2022": "Junior Year (2021-2022)",
            "2020-2021": "Sophomore Year (2020-2021)",
            "2019-2020": "Freshman Year (2019-2020)",
        }
        for season in season_order:
            st.write(season_labels.get(season, season))
            season_log = next((log for log in logs if log.season == season), None)
            if season_log is not None:
                st.dataframe(season_log.games, use_container_width=True)
            else:
                st.info("Game log not listed on the Deseret page for this season.")

        st.subheader("Football (2022 Season)")
        try:
            football_html = fetch_html(FOOTBALL_URL)
            football_stats = parse_football_receiver_stats(football_html)
            football_log = parse_football_game_log(football_html)
        except Exception as exc:
            st.error(f"Unable to load football data: {exc}")
            football_stats = {}
            football_log = None

        if show_diagnostics:
            st.subheader("Diagnostics: Football Tables")
            try:
                tables = _read_tables(football_html)
                with st.expander(f"Football Page ({len(tables)} tables)"):
                    for idx, table in enumerate(tables):
                        cols = [str(col) for col in table.columns]
                        st.write(f"Table {idx + 1} columns: {cols}")
            except Exception as exc:
                st.warning(f"Diagnostics failed: {exc}")

        if football_stats:
            for section, stats in football_stats.items():
                st.write(f"{section} (2022-2023)")
                st.dataframe(pd.DataFrame([stats]), use_container_width=True)
        else:
            st.info("No football season stats were found for 2022-2023.")

        st.subheader("Football Game Log (2022 Season)")
        if football_log is not None and not football_log.empty:
            st.dataframe(football_log, use_container_width=True)
        else:
            st.info("Football game log not found.")

        st.subheader("Player Breakdown")
        if per_game_rows:
            per_game_df = pd.DataFrame(per_game_rows).sort_values("Season")
            latest = per_game_df.iloc[-1]
            earliest = per_game_df.iloc[0]
            middle = per_game_df.iloc[len(per_game_df) // 2]
            paragraph = (
                "Sam Stevenson’s high school career shows steady, multi-year growth and a "
                "clear progression into a senior-year impact role. His production climbed "
                f"from {earliest['PPG']:.1f} PPG early in his career to {latest['PPG']:.1f} "
                f"PPG as a senior, while assists rose from {earliest['APG']:.1f} to "
                f"{latest['APG']:.1f} and rebounds from {earliest['RPG']:.1f} to "
                f"{latest['RPG']:.1f}, pointing to expanding responsibility and a more "
                "complete all-around game. The stat mix highlights a scoring guard who can "
                "space the floor (3PT volume), create for teammates, and contribute on the "
                "defensive end with steals and occasional blocks, which aligns with a "
                "two-way, combo-guard profile. By the middle of his career, he was already "
                f"posting {middle['PPG']:.1f} PPG, indicating the jump to senior production "
                "was the continuation of an upward trend rather than a single-year spike. "
                "Overall, the numbers reflect consistent year-over-year improvement, steady "
                "game-to-game contribution, and a dependable role as a primary or secondary "
                "creator."
            )
            st.write(paragraph)
            highlights = _best_games(logs, top_n=6)
            if highlights:
                st.write("Key performances:")
                for line in highlights:
                    st.write(f"- {line}")
        else:
            st.info("Not enough data to generate a detailed breakdown.")

    with tabs[1]:
        st.subheader("Opponent Comparison")
        opponent_df = build_opponent_games(logs)
        if opponent_df.empty:
            st.info("Opponent comparison data not available.")
        else:
            opponents = sorted(opponent_df["Opponent"].dropna().unique())
            selected_opponent = st.selectbox("Select an opponent", opponents)
            stat_order = ["PPG", "3PG", "RPG", "APG", "SPG", "BPG"]
            stat_selection = st.multiselect(
                "Select stats to compare by opponent",
                stat_order,
                default=["PPG", "RPG", "APG"],
            )
            if selected_opponent and stat_selection:
                filtered = opponent_df[
                    (opponent_df["Opponent"] == selected_opponent)
                    & opponent_df["Season"].notna()
                ].copy()
                if filtered.empty:
                    st.info("No games logged for this opponent across seasons.")
                else:
                    st.write(f"Opponent: {selected_opponent}")
                    st.dataframe(
                        filtered[
                            [
                                "Season",
                                "Date",
                                "Opponent",
                                "PPG",
                                "3PG",
                                "RPG",
                                "APG",
                                "SPG",
                                "BPG",
                            ]
                        ],
                        use_container_width=True,
                    )

                    season_order = ["2019-2020", "2020-2021", "2021-2022", "2022-2023"]
                    filtered["SeasonOrder"] = filtered["Season"].apply(
                        lambda s: season_order.index(s) if s in season_order else 99
                    )
                    filtered["DateSort"] = pd.to_datetime(
                        filtered["Date"], format="%m/%d", errors="coerce"
                    )
                    filtered = filtered.sort_values(
                        ["SeasonOrder", "DateSort"], na_position="last"
                    )
                    filtered["GameIndex"] = range(1, len(filtered) + 1)

                    long_df = filtered.melt(
                        id_vars=["Season", "Date", "Opponent", "GameIndex", "HomeAway"],
                        value_vars=stat_selection,
                        var_name="Stat",
                        value_name="Value",
                    )
                    base = alt.Chart(long_df).encode(
                        x=alt.X("GameIndex:Q", title="Games vs opponent (chronological)"),
                        y=alt.Y("Value:Q", title="Stat value"),
                        tooltip=[
                            "Season:N",
                            "Date:N",
                            "Stat:N",
                            "HomeAway:N",
                            alt.Tooltip("Value:Q", format=".2f"),
                        ],
                    )

                    stat_colors = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
                    line = (
                        base.mark_line(strokeWidth=2)
                        .encode(
                            color=alt.Color(
                                "Stat:N",
                                title="Stat",
                                scale=alt.Scale(domain=stat_order, range=stat_colors),
                            ),
                            detail=alt.Detail("Stat:N"),
                            order=alt.Order("GameIndex:Q"),
                        )
                        .transform_filter(alt.datum.Value != None)
                    )

                    points = base.mark_point(filled=True, size=70).encode(
                        color=alt.Color(
                            "HomeAway:N",
                            title="Home/Away",
                            scale=alt.Scale(domain=["Home", "Away"], range=["#2a6fdb", "#7db6ff"]),
                        ),
                        shape=alt.Shape("Stat:N", title="Stat"),
                    )

                    st.altair_chart(line + points, use_container_width=True)
            else:
                st.info("Select an opponent and at least one stat.")

        if selected_opponent:
            st.subheader("Opponent Summary")
            summaries = _summarize_opponents(
                opponent_df[opponent_df["Opponent"] == selected_opponent]
            )
            if summaries:
                for line in summaries:
                    st.write(line)
            else:
                st.info("Opponent summary not available.")

    with tabs[2]:
        st.subheader("David Stevenson Season Overview")

        david_profile = {}
        if david_totals:
            try:
                david_profile = parse_profile(fetch_html(DAVID_BASKETBALL_URL))
            except Exception:
                david_profile = {}

        if david_profile:
            cols = st.columns(4)
            for col, label in zip(cols, ["Class", "Height", "School", "Position"]):
                value = david_profile.get(label, "") or "—"
                col.metric(label, value)

        if david_totals:
            david_totals_df = pd.DataFrame([t.__dict__ for t in david_totals])
            st.write("Season totals")
            st.dataframe(david_totals_df, use_container_width=True)

        david_per_game_rows = [compute_per_game_from_totals(total) for total in david_totals]
        david_per_game_rows = [row for row in david_per_game_rows if row]

        if david_per_game_rows:
            david_per_game_df = pd.DataFrame(david_per_game_rows)
            st.write("Per-game averages (based on 23-game seasons)")
            st.dataframe(david_per_game_df, use_container_width=True)
            render_per_game_chart(david_per_game_df, key="david_overview")

            st.subheader("Improvement Highlights")
            for line in improvement_summary(david_per_game_df):
                st.write(f"- {line}")
        else:
            st.info("Season totals not found, so per-game averages are unavailable.")

        st.subheader("Basketball Game Logs By Season")
        season_order = ["2018-2019", "2017-2018", "2016-2017", "2015-2016"]
        season_labels = {
            "2018-2019": "Senior Year (2018-2019)",
            "2017-2018": "Junior Year (2017-2018)",
            "2016-2017": "Sophomore Year (2016-2017)",
            "2015-2016": "Freshman Year (2015-2016)",
        }
        for season in season_order:
            st.write(season_labels.get(season, season))
            season_log = next((log for log in david_logs if log.season == season), None)
            if season_log is not None:
                st.dataframe(season_log.games, use_container_width=True)
            else:
                st.info("Game log not listed on the Deseret page for this season.")

    with tabs[3]:
        st.subheader("David Opponent Comparison")
        david_opponent_df = build_opponent_games(david_logs)
        if david_opponent_df.empty:
            st.info("Opponent comparison data not available.")
        else:
            opponents = sorted(david_opponent_df["Opponent"].dropna().unique())
            selected_opponent = st.selectbox("Select an opponent", opponents, key="david_opponent")
            stat_order = ["PPG", "3PG", "RPG", "APG", "SPG", "BPG"]
            stat_selection = st.multiselect(
                "Select stats to compare by opponent",
                stat_order,
                default=["PPG", "RPG", "APG"],
                key="david_opponent_stats",
            )
            if selected_opponent and stat_selection:
                filtered = david_opponent_df[
                    (david_opponent_df["Opponent"] == selected_opponent)
                    & david_opponent_df["Season"].notna()
                ].copy()
                if filtered.empty:
                    st.info("No games logged for this opponent across seasons.")
                else:
                    st.write(f"Opponent: {selected_opponent}")
                    st.dataframe(
                        filtered[
                            [
                                "Season",
                                "Date",
                                "Opponent",
                                "PPG",
                                "3PG",
                                "RPG",
                                "APG",
                                "SPG",
                                "BPG",
                            ]
                        ],
                        use_container_width=True,
                    )

                    season_order = ["2015-2016", "2016-2017", "2017-2018", "2018-2019"]
                    filtered["SeasonOrder"] = filtered["Season"].apply(
                        lambda s: season_order.index(s) if s in season_order else 99
                    )
                    filtered["DateSort"] = pd.to_datetime(
                        filtered["Date"], format="%m/%d", errors="coerce"
                    )
                    filtered = filtered.sort_values(
                        ["SeasonOrder", "DateSort"], na_position="last"
                    )
                    filtered["GameIndex"] = range(1, len(filtered) + 1)

                    long_df = filtered.melt(
                        id_vars=["Season", "Date", "Opponent", "GameIndex", "HomeAway"],
                        value_vars=stat_selection,
                        var_name="Stat",
                        value_name="Value",
                    )
                    base = alt.Chart(long_df).encode(
                        x=alt.X("GameIndex:Q", title="Games vs opponent (chronological)"),
                        y=alt.Y("Value:Q", title="Stat value"),
                        tooltip=[
                            "Season:N",
                            "Date:N",
                            "Stat:N",
                            "HomeAway:N",
                            alt.Tooltip("Value:Q", format=".2f"),
                        ],
                    )

                    stat_colors = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
                    line = (
                        base.mark_line(strokeWidth=2)
                        .encode(
                            color=alt.Color(
                                "Stat:N",
                                title="Stat",
                                scale=alt.Scale(domain=stat_order, range=stat_colors),
                            ),
                            detail=alt.Detail("Stat:N"),
                            order=alt.Order("GameIndex:Q"),
                        )
                        .transform_filter(alt.datum.Value != None)
                    )

                    points = base.mark_point(filled=True, size=70).encode(
                        color=alt.Color(
                            "HomeAway:N",
                            title="Home/Away",
                            scale=alt.Scale(domain=["Home", "Away"], range=["#2a6fdb", "#7db6ff"]),
                        ),
                        shape=alt.Shape("Stat:N", title="Stat"),
                    )

                    st.altair_chart(line + points, use_container_width=True)
            else:
                st.info("Select an opponent and at least one stat.")

        if selected_opponent:
            st.subheader("Opponent Summary")
            summaries = _summarize_opponents(
                david_opponent_df[david_opponent_df["Opponent"] == selected_opponent]
            )
            if summaries:
                for line in summaries:
                    st.write(line)
            else:
                st.info("Opponent summary not available.")

    with tabs[4]:
        st.subheader("Sam vs David Stevenson (Junior & Senior)")

        sam_by_season = {log.season: log for log in logs}
        david_by_season = {log.season: log for log in david_logs}
        sam_totals_by_season = {t.season: t for t in totals}
        david_totals_by_season = {t.season: t for t in david_totals}

        comparison_pairs = [
            ("Junior Year (2021-2022)", "2021-2022", "2017-2018"),
            ("Senior Year (2022-2023)", "2022-2023", "2018-2019"),
        ]

        for label, sam_season, david_season in comparison_pairs:
            st.write(label)

            sam_log = sam_by_season.get(sam_season)
            david_log = david_by_season.get(david_season)

            cols = st.columns(2)
            with cols[0]:
                st.write(f"Sam ({sam_season})")
                if sam_log is not None:
                    st.dataframe(sam_log.games, use_container_width=True)
                else:
                    st.info("Game log not found.")
            with cols[1]:
                st.write(f"David ({david_season})")
                if david_log is not None:
                    st.dataframe(david_log.games, use_container_width=True)
                else:
                    st.info("Game log not found.")

            sam_avg = (
                compute_per_game_from_log(sam_log)
                if sam_log is not None
                else compute_per_game_from_totals(sam_totals_by_season.get(sam_season))
                if sam_totals_by_season.get(sam_season) is not None
                else {}
            )
            david_avg = (
                compute_per_game_from_log(david_log)
                if david_log is not None
                else compute_per_game_from_totals(david_totals_by_season.get(david_season))
                if david_totals_by_season.get(david_season) is not None
                else {}
            )

            if sam_avg and david_avg:
                avg_df = pd.DataFrame(
                    [
                        {"Player": "Sam", **sam_avg},
                        {"Player": "David", **david_avg},
                    ]
                )
                st.write("Per-game averages")
                st.dataframe(avg_df, use_container_width=True)

                stat_order = ["PPG", "3PG", "RPG", "APG", "SPG", "BPG"]
                long_avg = avg_df.melt(
                    id_vars=["Player"],
                    value_vars=stat_order,
                    var_name="Stat",
                    value_name="Value",
                )
                avg_chart = (
                    alt.Chart(long_avg)
                    .mark_bar()
                    .encode(
                        x=alt.X("Stat:N", sort=stat_order, title=None),
                        y=alt.Y("Value:Q", title="Per game"),
                        color=alt.Color("Player:N", title="Player"),
                        xOffset="Player:N",
                        tooltip=["Player:N", "Stat:N", alt.Tooltip("Value:Q", format=".2f")],
                    )
                )
                st.altair_chart(avg_chart, use_container_width=True)

                def _games_to_series(log: GameLog, player: str) -> pd.DataFrame:
                    df = log.games.copy()
                    for col in ["Points", "Rebounds", "Assists", "Steals", "Blocks", "Threes"]:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors="coerce")
                    df = df.dropna(subset=["Points"], how="all")
                    df["Player"] = player
                    df["GameIndex"] = range(1, len(df) + 1)
                    df.rename(
                        columns={
                            "Points": "PPG",
                            "Threes": "3PG",
                            "Rebounds": "RPG",
                            "Assists": "APG",
                            "Steals": "SPG",
                            "Blocks": "BPG",
                        },
                        inplace=True,
                    )
                    return df

                if sam_log is not None and david_log is not None:
                    sam_series = _games_to_series(sam_log, "Sam")
                    david_series = _games_to_series(david_log, "David")
                    series_df = pd.concat([sam_series, david_series], ignore_index=True)

                    stat_selection = st.multiselect(
                        f"Select stats to compare ({label})",
                        ["PPG", "3PG", "RPG", "APG", "SPG", "BPG"],
                        default=["PPG", "RPG", "APG"],
                        key=f"stats_{sam_season}",
                    )
                    if stat_selection:
                        long_series = series_df.melt(
                            id_vars=["Player", "GameIndex"],
                            value_vars=stat_selection,
                            var_name="Stat",
                            value_name="Value",
                        )
                        stat_colors = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
                        line_chart = (
                            alt.Chart(long_series)
                            .mark_line(point=True)
                            .encode(
                                x=alt.X("GameIndex:Q", title="Game (chronological)"),
                                y=alt.Y("Value:Q", title="Stat value"),
                                color=alt.Color(
                                    "Stat:N",
                                    title="Stat",
                                    scale=alt.Scale(domain=stat_order, range=stat_colors),
                                ),
                                strokeDash=alt.StrokeDash(
                                    "Player:N",
                                    title="Player",
                                    scale=alt.Scale(domain=["Sam", "David"], range=[(4, 2), (1, 0)]),
                                ),
                                detail=alt.Detail(["Player:N", "Stat:N"]),
                                tooltip=[
                                    "Player:N",
                                    "Stat:N",
                                    alt.Tooltip("Value:Q", format=".2f"),
                                ],
                            )
                        )
                        st.altair_chart(line_chart, use_container_width=True)
                    else:
                        st.info("Select at least one stat for the game-by-game comparison.")
                else:
                    st.info("Game-by-game comparison unavailable for this pairing.")

        st.subheader("Player Breakdown: Sam vs David")
        sam_highs = _career_highs(logs)
        david_highs = _career_highs(david_logs)
        if sam_highs and david_highs:
            breakdown = (
                "Both brothers show multi-category impact with scoring and playmaking, "
                "but the logs suggest differences in where they peaked. Sam’s career "
                f"highs include {sam_highs.get('PPG', {}).get('value', '—')} points, "
                f"{sam_highs.get('APG', {}).get('value', '—')} assists, and "
                f"{sam_highs.get('RPG', {}).get('value', '—')} rebounds, while David’s "
                f"top marks include {david_highs.get('PPG', {}).get('value', '—')} points, "
                f"{david_highs.get('APG', {}).get('value', '—')} assists, and "
                f"{david_highs.get('RPG', {}).get('value', '—')} rebounds. "
                "Sam’s game appears to tilt toward steady two-way contributions across "
                "steals and blocks, while David’s logs show his biggest nights coming "
                "through scoring and playmaking. The overlap in high-point outputs and "
                "assist totals suggests both were capable creators, with Sam showing a "
                "slightly more balanced defensive profile and David showing more "
                "peak scoring nights in his junior/senior seasons."
            )
            st.write(breakdown)
            st.write(
                "If they had shared the backcourt, their skill sets would have "
                "complemented well: David’s peak scoring and shot creation paired with "
                "Sam’s steady all-around production, defensive activity, and secondary "
                "playmaking. Together, that mix would likely have stretched defenses, "
                "kept pressure on the rim and perimeter, and created a dominant two-guard "
                "tandem capable of controlling tempo and closing games."
            )
        else:
            st.info("Not enough data to generate the sibling breakdown.")

    st.subheader("Sources")
    st.write(f"Basketball (Senior): {BASKETBALL_SENIOR_URL}")
    st.write(f"Basketball (Freshman-Sophomore-Junior): {BASKETBALL_UNDERCLASS_URL}")
    st.write(f"Football (2022): {FOOTBALL_URL}")


if __name__ == "__main__":
    main()
