import pandas as pd
import preprocessRawData
import visualization
import model

if __name__ == '__main__':
    # load data
    path = 'data/'
    scores = pd.read_csv(path + "spreadspoke_scores.csv")
    teams = pd.read_csv(path + "nfl_teams.csv")
    elo = pd.read_csv(path + "nfl_elo.csv")

    # load preprocessed data
    preprocess = pd.read_csv(path + "data_preprocessed.csv")

    # visualization
    visualization.visualize_raw(scores)
    visualization.visualize_processed(preprocess)
    visualization.print_factor(preprocess)

    # train model
    model.bestFit('data/data_preprocessed.csv')

