import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn import linear_model, datasets
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import RandomForestClassifier
from matplotlib.lines import Line2D
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix


model_list = {
    "L1_Logistic_Regression": linear_model.LogisticRegression(solver='liblinear'),
    "L2_Logistic_Regression": linear_model.LogisticRegression(solver='lbfgs'),
    "Random_Forest": RandomForestClassifier(),
    "LinearSVM": LinearSVC(),
    # "NuSVM": NuSVC(decision_function_shape='ovo')
}

params_list = {
    "L1_Logistic_Regression": {'C': [10**i for i in range(-5,5)]},
    "L2_Logistic_Regression":  {'C': [10**i for i in range(-5,5)]},
    "Random_Forest": {'n_estimators': [10*i for i in range(1,10)]},
    "LinearSVM": {'C': [10**i for i in range(-5,5)]},
    # "NuSVM": {'nu': np.arange(0.05,0.55,0.05)}
}


def grid_search(X, Y, m, cs, K):
    clf = GridSearchCV(m, cs, cv=K)
    clf.fit(X,Y)
    print clf.cv_results_
    return clf.cv_results_['mean_test_score']


def run_all_models(X_train, Y_train, X_test, Y_test, K=5):
    scores = {}
    for m in model_list:
        print ('run', m)
        scores[m] = grid_search(X_train, Y_train, model_list[m], params_list[m], K)

    print (scores)
    max_score = 0
    for model_name, score in scores.items():
        if np.max(score) > max_score:
            max_score = np.max(score)
            param_key = list(params_list[model_name])[0]
            optimal_params = { param_key: params_list[model_name][param_key][np.argmax(score)]}
            optimal_model = model_list[model_name]

    # Build Model (with whole data)
    optimal_model.set_params(**optimal_params)
    optimal_model.fit(X_train, Y_train)
    Y_pred = optimal_model.predict(X_test)

    # Evaluation
    print (confusion_matrix(Y_test, Y_pred))

    return scores


def boxPlot(data):

    tmp = []
    for v in data.values():
        tmp.append(v)

    fig = plt.figure()
    plt.figure(figsize=(10,6))

    ax = plt.subplot(111)
    for i in range(len(tmp)):
        ax.boxplot(tmp[i], positions = [i],widths = 0.35 ,showfliers=False, patch_artist=True)
        ax.set_title('Comparison of ML models accuracy', fontsize=20)

    plt.xticks([0, 1, 2, 3, 4], data.keys())
    ax.set_xlim(-1,5)
    plt.savefig('test.png')


if __name__ == "__main__":
    K = 5
    plotScore = run_all_models(Xtrain, Ytrain, Xtest, Ytest, K)
    boxPlot(plotScore)
    plt.show()
