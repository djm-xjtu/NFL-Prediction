import seaborn as sns


def visualize_raw(scores):
    # basic visualization of the raw datasets
    scores.head()
    scores.info()
    scores.describe().transpose()


def print_factor(scores):
    # print the factors for the preprocessed dataset
    home_win = "{:.2f}".format(
        (sum((scores.result == 1) & (scores.stadium_neutral == 0)) / sum(scores.stadium_neutral == 0)) * 100)
    away_win = "{:.2f}".format(
        (sum((scores.result == 0) & (scores.stadium_neutral == 0)) / sum(scores.stadium_neutral == 0)) * 100)
    under_line = "{:.2f}".format(
        (sum((scores.score_home + scores.score_away) < scores.over_under_line) / len(scores)) * 100)
    over_line = "{:.2f}".format(
        (sum((scores.score_home + scores.score_away) > scores.over_under_line) / len(scores)) * 100)
    equal_line = "{:.2f}".format(
        (sum((scores.score_home + scores.score_away) == scores.over_under_line) / len(scores)) * 100)
    favored = "{:.2f}".format(
        (sum(((scores.welcome_home == 1) & (scores.result == 1)) | ((scores.welcome_away == 1) & (scores.result == 0)))
         / len(scores)) * 100)
    cover = "{:.2f}".format(
        (sum(((scores.welcome_home == 1) & ((scores.score_away - scores.score_home) < scores.spread_favorite)) |
             ((scores.welcome_away == 1) & ((
                                                        scores.score_home - scores.score_away) < scores.spread_favorite)))  # use score_home - score_away because the fav are swap
         / len(scores)) * 100)
    ats = "{:.2f}".format(
        (sum(((scores.welcome_home == 1) & ((scores.score_away - scores.score_home) > scores.spread_favorite)) |
             ((scores.welcome_away == 1) & ((scores.score_home - scores.score_away) > scores.spread_favorite)))
         / len(scores)) * 100)
    # print all percentages
    print("Number of Games: " + str(len(scores)))
    print("Home Straight Up Win Percentage: " + home_win + "%")
    print("Away Straight Up Win Percentage: " + away_win + "%")
    print("Under Average Percentage: " + under_line + "%")
    print("Over Average Percentage: " + over_line + "%")
    print("Equal Average Percentage: " + equal_line + "%")
    print("Favored Win Percentage: " + favored + "%")
    print("Cover The Spread Percentage: " + cover + "%")
    print("Against The Spread Percentage: " + ats + "%")


def visualize_processed(scores):
    # explanatory data analysis
    scores.columns.values
    scores.describe().transpose()
    g = sns.FacetGrid(scores, col="welomce_home", size=5)
    g.map(sns.barplot, "result")
    print(scores[['welcome_home', 'result']].groupby(['welcome_home'], as_index=False).mean())
    sns.violinplot(x="result", y="team_home_currentSeason_win_predict", data=scores)
    sns.violinplot(x="result", y="team_home_lastSeason_win_predict", data=scores)
    sns.pairplot(scores[['elo_prob1',
                         'team_away_currentSeason_win_predict', 'team_home_currentSeason_win_predict',
                         'team_home_lastSeason_win_predict', 'team_away_lastSeason_win_predict', 'result']],
                 hue='result', height=2.5)
    scores.summary
