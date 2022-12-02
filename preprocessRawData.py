import numpy as np
import pandas as pd


def preprocess(scores, teams, elo):
    # preprocess / clean RAW DATA
    path = 'data/'
    # replace null with nan
    scores = scores.replace(r'^\s*$', np.nan, regex=True)
    scores = scores[scores.schedule_season >= 2000]

    scores.reset_index(drop=True, inplace=True)
    scores['over_under_line'] = scores.over_under_line.astype(float)

    # transfer full team name into shortname
    scores['team_home'] = scores.team_home.map(teams.set_index('team_name')['team_id'].to_dict())
    scores['team_away'] = scores.team_away.map(teams.set_index('team_name')['team_id'].to_dict())

    # add 'home_welcome' and 'away_welcome' columns
    scores.loc[scores.team_favorite_id == scores.team_home, 'home_welcome'] = 1
    scores.loc[scores.team_favorite_id == scores.team_away, 'away_welcome'] = 1
    scores.home_welcome.fillna(0, inplace=True)
    scores.away_welcome.fillna(0, inplace=True)

    # add 'over' column
    scores.loc[((scores.score_home + scores.score_away) > scores.over_under_line), 'over'] = 1
    scores.over.fillna(0, inplace=True)

    # change stadium neutral and schedule playoff from boolean into integer
    scores['stadium_neutral'] = scores.stadium_neutral.astype(int)
    scores['schedule_playoff'] = scores.schedule_playoff.astype(int)

    # change data type of date columns
    scores['schedule_date'] = pd.to_datetime(scores['schedule_date'])
    elo['date'] = pd.to_datetime(elo['date'])

    # fix some string type schedule_week
    scores.loc[(scores.schedule_week == 'Wildcard') | (scores.schedule_week == 'WildCard'), 'schedule_week'] = '18'
    scores.loc[(scores.schedule_week == 'Division'), 'schedule_week'] = '19'
    scores.loc[(scores.schedule_week == 'Conference'), 'schedule_week'] = '20'
    scores.loc[(scores.schedule_week == 'Superbowl') | (scores.schedule_week == 'SuperBowl'), 'schedule_week'] = '21'
    scores['schedule_week'] = scores.schedule_week.astype(int)

    # uniform Washington's shortname
    elo.loc[elo.team1 == 'WSH', 'team1'] = 'WAS'
    elo.loc[elo.team2 == 'WSH', 'team2'] = 'WAS'

    # add team_home_currentSeason_win_predict, team_away_currentSeason_win_predict
    # add team_home_lastSeason_win_predict,    team_away_lastSeason_win_predict
    for team in teams.team_id.unique().tolist():
        for season in range(2000, 2022):
            wins, games_played = 0., 0.
            for week in range(1, 18):
                current_game = scores[
                    ((scores.team_home == team) | (scores.team_away == team)) & (scores.schedule_season == season) & (
                            scores.schedule_week == week)]
                # If a game exists
                if (current_game.shape[0] == 1):
                    current_game = current_game.iloc[0]

                    if ((current_game.team_home == team) & (current_game.score_home > current_game.score_away)):
                        wins += 1

                    elif ((current_game.team_away == team) & (current_game.score_away > current_game.score_home)):
                        wins += 1

                    if (current_game.score_away != current_game.score_home):
                        games_played += 1

                    # If week one put default record as 0
                    if (week == 1):
                        if (current_game.team_home == team):
                            scores.loc[(scores.team_home == team) & (scores.schedule_season == season) & (
                                    scores.schedule_week == week), 'team_home_currentSeason_win_predict'] = 0
                        else:
                            scores.loc[(scores.team_away == team) & (scores.schedule_season == season) & (
                                    scores.schedule_week == week), 'team_away_currentSeason_win_predict'] = 0

                next_week_game = scores[
                    ((scores.team_home == team) | (scores.team_away == team)) & (scores.schedule_season == season) & (
                            scores.schedule_week == week + 1)]
                if (next_week_game.shape[0] == 1):
                    next_week_game = next_week_game.iloc[0]
                    if (next_week_game.team_home == team):
                        scores.loc[(scores.team_home == team) & (scores.schedule_season == season) & (
                                scores.schedule_week == week + 1), 'team_home_currentSeason_win_predict'] = 0 if games_played == 0 else wins / games_played
                    else:
                        scores.loc[(scores.team_away == team) & (scores.schedule_season == season) & (
                                scores.schedule_week == week + 1), 'team_away_currentSeason_win_predict'] = 0 if games_played == 0 else wins / games_played
                else:
                    next_twoweek_game = scores[
                        ((scores.team_home == team) | (scores.team_away == team)) & (
                                scores.schedule_season == season) & (
                                scores.schedule_week == week + 2)]
                    if (next_twoweek_game.shape[0] == 1):
                        next_twoweek_game = next_twoweek_game.iloc[0]
                        if (next_twoweek_game.team_home == team):
                            scores.loc[(scores.team_home == team) & (scores.schedule_season == season) & (
                                    scores.schedule_week == week + 2), 'team_home_currentSeason_win_predict'] = 0 if games_played == 0 else wins / games_played
                        else:
                            scores.loc[(scores.team_away == team) & (scores.schedule_season == season) & (
                                    scores.schedule_week == week + 2), 'team_away_currentSeason_win_predict'] = 0 if games_played == 0 else wins / games_played

            # if beyond week 17 (playoffs use season record)
            for postseason_week in range(18, 22):
                current_game = scores[
                    ((scores.team_home == team) | (scores.team_away == team)) & (scores.schedule_season == season) & (
                            scores.schedule_week == postseason_week)]
                if (current_game.shape[0] == 1):
                    current_game = current_game.iloc[0]
                    if (current_game.team_home == team):
                        scores.loc[(scores.team_home == team) & (scores.schedule_season == season) & (
                                scores.schedule_week == postseason_week), 'team_home_currentSeason_win_predict'] = 0 if games_played == 0 else wins / games_played
                    else:
                        scores.loc[(scores.team_away == team) & (scores.schedule_season == season) & (
                                scores.schedule_week == postseason_week), 'team_away_currentSeason_win_predict'] = 0 if games_played == 0 else wins / games_played

            next_season = season + 1
            for week in range(1, 22):
                next_season_game = scores[
                    ((scores.team_home == team) | (scores.team_away == team)) & (
                            scores.schedule_season == next_season) & (
                            scores.schedule_week == week)]
                if (next_season_game.shape[0] == 1):
                    next_season_game = next_season_game.iloc[0]
                    if (next_season_game.team_home == team):
                        scores.loc[(scores.team_home == team) & (scores.schedule_season == next_season) & (
                                scores.schedule_week == week), 'team_home_lastSeason_win_predict'] = 0 if games_played == 0 else wins / games_played
                    elif (next_season_game.team_away == team):
                        scores.loc[(scores.team_away == team) & (scores.schedule_season == next_season) & (
                                scores.schedule_week == week), 'team_away_lastSeason_win_predict'] = 0 if games_played == 0 else wins / games_played

    # add division_game column
    scores = pd.merge(scores, teams[["team_id", "team_division"]], how='inner', left_on=['team_home'],
                      right_on=['team_id'])
    scores = pd.merge(scores, teams[["team_id", "team_division"]], how='inner', left_on=['team_away'],
                      right_on=['team_id'])
    scores = scores.rename(columns={'team_division_x': 'team_home_division', 'team_division_y': 'team_away_division'})

    # delete unused columns
    scores = scores.drop(['team_id_x', 'team_id_y'], axis=1)
    scores["division_game"] = scores["team_home_division"] == scores["team_away_division"]

    # merge data into one dataframe
    scores = scores.merge(elo[['date', 'team1', 'team2', 'elo_prob1', 'elo_prob2']],
                          left_on=['schedule_date', 'team_home', 'team_away'], right_on=['date', 'team1', 'team2'],
                          how='left')

    elo2 = elo.rename(columns={'team1': 'team2',
                               'team2': 'team1',
                               'elo1': 'elo2',
                               'elo2': 'elo1',
                               'elo_prob1': 'elo_prob2',
                               'elo_prob2': 'elo_prob1'})

    scores = scores.merge(elo2[['date', 'team1', 'team2', 'elo_prob1', 'elo_prob2']],
                          left_on=['schedule_date', 'team_home', 'team_away'], right_on=['date', 'team1', 'team2'],
                          how='left')

    # separating merged columns into x and y cols
    x_cols = ['date_x', 'team1_x', 'team2_x', 'elo_prob1_x', 'elo_prob2_x']
    y_cols = ['date_y', 'team1_y', 'team2_y', 'elo_prob1_y', 'elo_prob2_y']

    # filling null values for games_elo merged cols
    for x, y in zip(x_cols, y_cols):
        scores[x] = scores[x].fillna(scores[y])

    scores = scores[['schedule_date', 'schedule_season', 'schedule_week',
             'schedule_playoff', 'team_home', 'score_home', 'score_away',
             'team_away', 'team_favorite_id', 'spread_favorite',
             'over_under_line', 'stadium', 'stadium_neutral',
             'weather_temperature', 'weather_wind_mph', 'weather_humidity',
             'weather_detail', 'home_favorite', 'away_favorite', 'over',
             'team_away_current_win_pct', 'team_home_current_win_pct',
             'team_home_lastseason_win_pct', 'team_away_lastseason_win_pct',
             'team_home_division', 'team_away_division', 'division_game',
             'elo_prob1_x', 'elo_prob2_x']]

    # remove _x ending from column names
    scores.columns = scores.columns.str.replace('_x', '')

    scores = scores.dropna(subset=['elo_prob1', 'elo_prob2'])
    scores['result'] = (scores.score_home > scores.score_away).astype(int)
    scores.loc[scores.team_favorite_id == scores.team_home, 'home_favorite'] = 1
    scores.loc[scores.team_favorite_id == scores.team_away, 'away_favorite'] = 1
    scores.home_favorite.fillna(0, inplace=True)
    scores.away_favorite.fillna(0, inplace=True)
    scores.division_game = scores.division_game.astype(int)
    scores.home_favorite = scores.home_favorite.astype(int)
    scores.away_favorite = scores.away_favorite.astype(int)

    # convert into csv file
    scores.to_csv(path + "data_preprocessed.csv", index=False)

    
if __name__ == '__main__':
    # load data
    path = 'data/'
    scores = pd.read_csv(path + "spreadspoke_scores.csv")
    teams = pd.read_csv(path + "nfl_teams.csv")
    elo = pd.read_csv(path + "nfl_elo.csv")
    preprocess(scores, teams, elo)

