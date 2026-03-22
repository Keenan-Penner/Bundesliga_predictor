import pandas as pd 
import requests
import numpy as np
from collections import defaultdict
from typing import Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from rapidfuzz import process, fuzz

def get_BL_matches(season: int) -> pd.DataFrame :
    url = f"https://raw.githubusercontent.com/openfootball/football.json/master/{season}-{season%100 + 1}/de.1.json"
    data = requests.get(url).json()
    rows = []
    for match in data["matches"]:
        rows.append({
            "matchday": match["round"],
            "date": match["date"],
            "home": match["team1"],
            "away": match["team2"],
            "score_home": match["score"]["ft"][0],
            "score_away": match["score"]["ft"][1]
        })
    matches = pd.DataFrame(rows)
    return matches

#creates the bundesliga table for the season
def create_table(season: int) -> pd.DataFrame:
    season_matches = get_BL_matches(season)
    teams: Dict[str, Dict[str, int]] = defaultdict(lambda: {
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "goals_for": 0,
            "goals_against": 0,
            "home_wins": 0,
            "home_losses": 0
        })
    for _, row in season_matches.iterrows():
        home = row["home"]
        away = row["away"]
        home_goals = row["score_home"]
        away_goals = row["score_away"]
        teams[home]["goals_for"] += home_goals
        teams[home]["goals_against"] += away_goals
        teams[away]["goals_for"] += away_goals
        teams[away]["goals_against"] += home_goals
        if home_goals > away_goals:
            teams[home]["wins"] += 1
            teams[home]["home_wins"] += 1
            teams[away]["losses"] += 1
        elif home_goals < away_goals:
            teams[away]["wins"] += 1
            teams[home]["home_losses"] += 1
            teams[home]["losses"] += 1
        else:
            teams[away]["draws"] += 1
            teams[home]["draws"] += 1
    data = []
    for team, statistics in teams.items():
        goal_diff = statistics["goals_for"] - statistics["goals_against"]
        points = statistics["wins"] * 3 + statistics["draws"] 
        data.append({
            "team_name" : team,
            "points" : points,
            "win" : statistics["wins"],
            "draw" : statistics["draws"],
            "loss" : statistics["losses"],
            "goals_for" : statistics["goals_for"],
            "goals_against" : statistics["goals_against"],
            "goal_diff" : goal_diff,
            "home_wins" : statistics["home_wins"],
            "home_losses" : statistics["home_losses"]
        })
    table = pd.DataFrame(data)
    table = table.sort_values(["points", "goal_diff", "goals_for"], ascending= [False, False, False]).reset_index(drop= True)
    table["position"] = table.index + 1
    return table

def transform_value(col):
    if "m" in col:
        col = float(col.replace("m", "")) * 1000000
        return col
    elif "k" in col:
        col = float(col.replace("k", "")) * 1000
        return col
    return col

# useful functions for data cleaning
def adjust_values(stats : pd.DataFrame) -> pd.DataFrame :
    # clean data
    stats["value"] = stats["value"].astype(str).str.replace("€", "")
    stats["value"] = stats["value"].astype(str).str.replace(",", ".")
    stats["value"] = stats["value"].apply(transform_value)
    #make value relative
    total_budget = stats["value"].sum()
    stats["rel_value"] = stats["value"] / total_budget
    #get avg player value
    stats["avg_player_value"] = stats["value"] / stats["size"]
    #drop uninteresting columns
    stats.drop(columns= "foreigners", inplace= True)
    stats.drop(columns= "size", inplace= True)
    stats.drop(columns= "value", inplace= True)
    return stats

#merge season tables and pre season stats on club name (requires matching club names)
def add_pre_season_stats(season : int, table : pd.DataFrame, key_left : str) -> pd.DataFrame :
    stats = pd.read_csv(f"data/squad_values_{season}-{season +1}.csv")
    #fuzzy matching - the names from the table are not necessarily the same as the ones in the stats
    teams_left = table[key_left].tolist()

    def match_team(teams_right):
        match, score, _ = process.extractOne(
            teams_right,
            teams_left,
            scorer = fuzz.token_sort_ratio
        )
        return match if score > 60 else None
    # print(f"before: {stats}")
    stats["team"] = stats["team"].apply(match_team)
    # print(f"after: {stats}")
    result = pd.merge(table, stats, left_on= key_left, right_on="team", how= "left")
    result.drop(columns="team", inplace= True)
    return result

# We can use the fact that we will be using features from previous seasons along with features from before the season start

LAST_SEASON_FEATURES = ["points", "win", "draw", "loss", "goals_for", 
                         "goals_against", "goal_diff", "home_wins", "home_losses"]
PRESEASON_FEATURES = ["value", "size", "age"]

def prep_data2(start_date : int):
    # we prepare the training data for seasons prior to the current season
    season_tables : Dict[str, pd.DataFrame] = {}
    for year in range(start_date, 2025):
        table = create_table(year)
        season_tables[f"{year}"] = table
    feature_rows = []
    target_rows = []
    for i in range(start_date, 2024):
        prev_table = season_tables[f"{i}"].copy()
        curr_table = season_tables[f"{i +1}"].copy()
        prev_table[LAST_SEASON_FEATURES] = prev_table[LAST_SEASON_FEATURES].astype(float)
        curr_table[LAST_SEASON_FEATURES] = curr_table[LAST_SEASON_FEATURES].astype(float)
        prev_stats = pd.read_csv(f"data/squad_values_{i}-{i+1}.csv").set_index("team")
        curr_stats = pd.read_csv(f"data/squad_values_{i+1}-{i+2}.csv").set_index("team")
        prev_teams = set(prev_stats.index)
        curr_teams = set(curr_stats.index)
        promoted_teams = curr_teams - prev_teams
        promoted_teams = list(promoted_teams)
        n_promoted = len(promoted_teams)
        prev_table.iloc[-n_promoted:, prev_table.columns.isin(LAST_SEASON_FEATURES)] = prev_table.tail(n_promoted)[LAST_SEASON_FEATURES].mean().values 
        #we change the names of the teams in the last two rows to fit the names of the newly promoted teams
        for index, team in enumerate(promoted_teams):
            prev_table.iloc[[-n_promoted + index], prev_table.columns.get_loc("team_name")] = promoted_teams[index]
        prev_table = add_pre_season_stats(i+1, prev_table, "team_name")
        prev_table = adjust_values(prev_table)
        prev_table = prev_table.set_index("team_name")
        # Append features (all columns except position)
        feature_rows.append(prev_table.drop(columns=["position"]))
        curr_table = curr_table.set_index("team_name")
        target_rows.append(curr_table["position"])  # position in the NEXT season is the target
    X_train = pd.concat(feature_rows, axis=0).reset_index(drop=True)
    y_train = pd.concat(target_rows, axis=0).reset_index(drop=True)
    last_table = create_table(2024)
    last_table[LAST_SEASON_FEATURES] = last_table[LAST_SEASON_FEATURES].astype(float) # we will be taking avg, so no longer integer valued
    #since we dont actually have stats for prmoted teams, we take avg stats of the relegated teams
    last_table.iloc[-2:, last_table.columns.isin(LAST_SEASON_FEATURES)] = last_table.tail(2)[LAST_SEASON_FEATURES].mean().values 
    #we change the names of the teams in the last two rows to fit the names of the newly promoted teams
    last_table.loc[last_table["team_name"] == "Holstein Kiel", "team_name"] = "FC Köln"
    last_table.loc[last_table["team_name"] == "VfL Bochum 1848", "team_name"] = "Hamburger SV"
    last_table = add_pre_season_stats(2025, last_table, "team_name") #we are adding the pre_season stats for the following season, hence the 2025
    last_table = adjust_values(last_table)
    latest_features_df = last_table.set_index("team_name")
    latest_features_df = latest_features_df.drop(columns= ["position"])
    return X_train, y_train, latest_features_df

#model setup
def train_model(X, y) -> Pipeline: 
    model = Pipeline([
        ("rf", RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42,
        ))
    ])
    model.fit(X, y)
    return model

def main():
    table = {}
    X, y ,df = prep_data2(2019)
    model = train_model(X, y)
    prob = model.predict_proba(df)
    classes = model.named_steps["rf"].classes_
    exp_positions = prob.dot(classes)
    prediction_df = pd.DataFrame({
            "team": df.index,
            "expected_position": exp_positions
            })
    prediction_df.sort_values(["expected_position"], ascending=True, inplace= True)
    prediction_df
    prediction_df = prediction_df.reset_index(drop= True)
    prediction_df
    prediction_df["position"] = prediction_df.index +1
    table["standings"] = [prediction_df]
    prob = model.predict_proba(df)
    classes = model.named_steps["rf"].classes_
    # positions
    zones = {
        "Champion":      classes <= 1,
        "2_4":      (classes >= 2) & (classes <= 4),
        "5_10":   (classes >= 5) & (classes <= 10),
        "11_17": (classes >= 11) & (classes <= 17),
        "Relegated": classes >= 18
    }
    prediction_df = pd.DataFrame({"team": df.index})
    prediction_df["expected_position"] = prob.dot(classes)
    for zone, mask in zones.items():
        prediction_df[f"prob_{zone}"] = prob[:, mask].sum(axis=1)
    prediction_df = prediction_df.sort_values("expected_position")
    table["standings"].append(prediction_df)
    print(table["standings"][0])
    print(table["standings"][1])    

if __name__ == "__main__":
    main()