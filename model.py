# import matplotlib.pyplot as plt
import numpy as np
import time

from sklearn.svm import SVC, LinearSVC
from sklearn import linear_model, datasets
from sklearn.ensemble import RandomForestClassifier

from spark_sklearn import GridSearchCV as SPGridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, roc_auc_score, cohen_kappa_score, make_scorer


model_list = {
    "RBF-SVM": SVC(kernel='rbf', decision_function_shape='ovr', class_weight='balanced'),
    "L1_Logistic_Regression": linear_model.LogisticRegression(solver='liblinear', class_weight='balanced'),
    "L2_Logistic_Regression": linear_model.LogisticRegression(solver='lbfgs', class_weight='balanced'),
    # "Random_Forest": RandomForestClassifier(class_weight='balanced'),
    "LinearSVM": LinearSVC(class_weight='balanced'),
}

params_list = {
    "RBF-SVM": {'C': [10**i for i in range(-5,5)]},
    "L1_Logistic_Regression": {'C': [10**i for i in range(-5,5)]},
    "L2_Logistic_Regression":  {'C': [10**i for i in range(-5,5)]},
    # "Random_Forest": {'n_estimators': [10], 'max_depth': [20,30], 'oob_score': [True]},
    # "Random_Forest": {'n_estimators': [50], 'max_depth': range(20,25), 'min_samples_leaf': [2**i for i in range(10,14)]},
    "LinearSVM": {'C': [10**i for i in range(-5,5)]},
}


def grid_search(sc, X, Y, m, cs, K):
    kappa_scorer = make_scorer(cohen_kappa_score)
    clf = SPGridSearchCV(sc, m, cs, cv=K, scoring=kappa_scorer)
    clf.fit(X,Y)
    print (clf.cv_results_)
    return clf.cv_results_['mean_test_score']


def run_all_models(sc, X_train, Y_train, X_test, Y_test, K=5):
    scores = {}
    for m in model_list:
        print ('run', m)
        scores[m] = grid_search(sc, X_train, Y_train, model_list[m], params_list[m], K)

    print (scores)
    max_score = 0
    for model_name, score in scores.items():
        if np.max(score) > max_score:
            max_score = np.max(score)
            param_key = list(params_list[model_name])[0]
            optimal_params = {param_key: params_list[model_name][param_key][np.argmax(score)]}
            optimal_model = model_list[model_name]

    # Build Model (with whole data)
    optimal_model.set_params(**optimal_params)
    optimal_model.fit(X_train, Y_train)
    Y_pred = optimal_model.predict(X_test)

    # Evaluation
    print ('confusion_matrix', confusion_matrix(Y_test, Y_pred))
    print ('roc_auc_score', roc_auc_score(Y_test, Y_pred))
    print ('cohen_kappa_score', cohen_kappa_score(Y_test, Y_pred))

    return scores


# def boxPlot(data):

#     tmp = []
#     for v in data.values():
#         tmp.append(v)

#     plt.figure(figsize=(10, 6))

#     ax = plt.subplot(111)
#     for i in range(len(tmp)):
#         # ax.boxplot(tmp[i], positions = [i],widths = 0.35 ,showfliers=False, patch_artist=True)
#         ax.boxplot(tmp[i], positions=[i], widths=0.35, patch_artist=True)
#         ax.set_title('Comparison of ML models accuracy', fontsize=20)

#     plt.xticks(range(len(tmp)), data.keys())
#     ax.set_xlim(-1, len(tmp))
#     fig_name = str(int(time.time())) + '.png'
#     plt.savefig(fig_name)
#     plt.show()

