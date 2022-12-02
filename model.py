import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)


def bestFit(filename):
    path = "data/"
    df = pd.read_csv(path + filename)
    train = df.copy()
    test = df.copy()
    train = train.loc[train['schedule_season'] < 2017]
    X_train = train[
        ['schedule_week', 'spread_favorite', 'over_under_line', 'home_favorite', 'team_away_current_win_pct',
         'team_home_current_win_pct',
         'team_home_lastseason_win_pct', 'team_away_lastseason_win_pct', 'division_game', 'elo_prob1']]
    y_train = train['result']

    model = []
    bst = xgb.XGBClassifier()
    lrg = LogisticRegression(solver='liblinear')
    dtc = DecisionTreeClassifier(max_depth=5, criterion='entropy')
    model.append(('LRG', LogisticRegression(solver='liblinear', max_iter=250)))
    model.append(('KNB', KNeighborsClassifier()))
    model.append(('GNB', GaussianNB()))
    model.append(('RFC', RandomForestClassifier(random_state=0, n_estimators=100)))
    model.append(('DTC', DecisionTreeClassifier(random_state=0, criterion='entropy', max_depth=5)))
    model.append(('SVC', LinearSVC(random_state=0)))
    model.append(('VOTE', VotingClassifier(estimators=[('boost', bst), ('dtc', dtc), ('lrg', lrg)], voting='soft')))

    clean_models = []
    for n, m in model:
        k_fold = model_selection.KFold(n_splits=5)
        result = model_selection.cross_val_score(m, X_train, y_train, cv=k_fold, scoring='roc_auc')
        clean_models.append([n, "%f" % result.mean()])
    clean_models = sorted(clean_models, key=(lambda x: x[1]), reverse=True)
    # Selected the best fit model
    for score in clean_models:
        print(score)
    print("The best fit model is: " + clean_models[0][0] + "\n" + "The score is: " + clean_models[0][1])


# run the function
bestFit("data_preprocessed.csv")

# Training the model:
path = "data/"
df = pd.read_csv(path + "data_preprocessed.csv")
train = df.copy()
test = df.copy()
train = train.loc[train['schedule_season'] < 2017]
test = test.loc[test['schedule_season'] > 2016]
X_train = train[
    ['schedule_week', 'spread_favorite', 'over_under_line', 'home_favorite', 'team_away_current_win_pct',
     'team_home_current_win_pct',
     'team_home_lastseason_win_pct', 'team_away_lastseason_win_pct', 'division_game', 'elo_prob1']]
y_train = train['result']

X_test = test[['schedule_week', 'spread_favorite', 'over_under_line', 'home_favorite', 'team_away_current_win_pct',
               'team_home_current_win_pct',
               'team_home_lastseason_win_pct', 'team_away_lastseason_win_pct', 'division_game', 'elo_prob1']]
y_test = test['result']

logist = LogisticRegression(solver='liblinear', max_iter=250)
logist.fit(X_train, y_train)
y_prediction = logist.predict(X_test)
y_predicted = logist.predict_proba(X_test)[:, 1]

# Find the importance of the feature
target_names = ['class 0', 'class 1']
print(classification_report(y_test, y_prediction, target_names=target_names))

# Draw the Confusion matrix
mat = metrics.confusion_matrix(y_test, y_prediction)
score = logist.score(X_test, y_test)
plt.figure(figsize=(6.5, 5))
sns.heatmap(mat, annot=True, fmt="g")
plt.title('LogisticRegression \nAccuracy:{0:.2f}'.format(score))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Finding the fpr and tpr for the threasholds of the classification
fpr, tpr, threshold = roc_curve(y_test, y_prediction)
rocAuc = auc(fpr, tpr)

plt.title('Receiver Operating features')
plt.plot(fpr, tpr, 'r', label='AUC = %0.2f' % rocAuc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'g--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False')
plt.ylabel('True')
plt.show()

# Comparing the model accuracy with elo evaluation
print("Metrics" + "\t\t" + "LRG Model" + "\t" + "Elo_prob1 Result")
print("ROC_AUC Score: " + "\t" + "{:.5f}".format(roc_auc_score(y_test, y_predicted)) + "\t\t" + "{:.5f}".format(
    roc_auc_score(test.result, test.elo_prob1)))
print("Brier Score: " + "\t" + "{:.5f}".format(brier_score_loss(y_test, y_predicted)) + "\t\t" + "{:.5f}".format(
    brier_score_loss(test.result, test.elo_prob1)))

# Simulations

test.loc[:, 'hm_prob'] = y_predicted
test = test[['schedule_season', 'schedule_week', 'team_home', 'team_away', 'elo_prob1', 'hm_prob', 'result']]
test['my_bet_won'] = (
            ((test.hm_prob <= 0.40) & (test.result == 0)) | ((test.hm_prob >= 0.60) & (test.result == 1))).astype(int)
test['elo_bet_won'] = (
            ((test.elo_prob1 >= 0.60) & (test.result == 1)) | ((test.elo_prob1 <= 0.40) & (test.result == 0))).astype(
    int)
test['my_bet_lost'] = (
            ((test.hm_prob <= 0.40) & (test.result == 1)) | ((test.hm_prob >= 0.60) & (test.result == 0))).astype(int)
test['elo_bet_lost'] = (
            ((test.elo_prob1 >= 0.60) & (test.result == 0)) | ((test.elo_prob1 <= 0.40) & (test.result == 1))).astype(
    int)
print("Possible Games: " + str(len(test)))
print("LRG model Win Percentage: " + "{:.4f}".format(
    test.my_bet_won.sum() / (test.my_bet_lost.sum() + test.my_bet_won.sum())))
print("Number of Bets Won: " + str(test.my_bet_won.sum()))
print("Number of Bets Made: " + str((test.my_bet_lost.sum() + test.my_bet_won.sum())))

print("Possible Games: " + str(len(test)))
print("LRG model Win Percentage: " + "{:.4f}".format(
    test.elo_bet_won.sum() / (test.elo_bet_lost.sum() + test.elo_bet_won.sum())))
print("Number of Bets Won: " + str(test.elo_bet_won.sum()))
print("Number of Bets Made: " + str((test.elo_bet_lost.sum() + test.elo_bet_won.sum())))
