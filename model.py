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
"L1_Logistic_Regression": linear_model.LogisticRegression(solver=liblinear),
"L2_Logistic_Regression": linear_model.LogisticRegression(solver=lbfgs),
"Random_Forest": RandomForestClassifier(),
"LinearSVM": LinearSVC(),
"NuSVM": NuSVC(decision_function_shape='ovo')
}

params_list = {
"L1_Logistic_Regression": {'C': [10**i for i in range(-5,5)]},
"L2_Logistic_Regression":  {'C': [10**i for i in range(-5,5)]},
"Random_Forest": ,
"LinearSVM": ,
"NuSVM":
}

def grid_search(X, Y, m, cs, K):
    clf = GridSearchCV(m, cs, cv=K)
    clf.fit(X,Y)
    return clf.cv_results_['mean_test_score']

def L1logisticReg(X,Y):
    score = {}
    #Regularisation strength, the smaller the stronger regularisation
    cs = [10**i for i in range(-5,5)]
    for c in cs:
        # Parameter c specifies regularisation strength, the smaller the stronger strength
        clf = linear_model.LogisticRegression(C = c, solver='liblinear')
        clf.fit(X, Y)
        score[c]=cross_val_score(clf, X, Y, scoring="accuracy", cv=5)
    return score


def L2logisticReg(X,Y):
    score = {}
    #Regularisation strength, the smaller the stronger regularisation
    cs = [10**i for i in range(-5,5)]
    for c in cs:
        # Parameter c specifies regularisation strength, the smaller the stronger strength
        clf = linear_model.LogisticRegression(C = c, solver='lbfgs')
        clf.fit(X, Y)
        score[c]=cross_val_score(clf, X, Y, scoring="accuracy", cv=5)
    return score


def randomForest(X, Y):
    score = {}
    cs = [10**i for i in range(1,5)]
    for c in cs:
        #n_estimators is The number of trees in the forest.
        clf = RandomForestClassifier(n_estimators=c)
        clf.fit(X, Y)
        score[c]=cross_val_score(clf, X, Y, scoring="accuracy", cv=5)
    return score


def linearSVM(X, Y):
    score = {}
    cs = [10**i for i in range(1,5)]
    for c in cs:
        #Penalty parameter C of the error term
        clf = LinearSVC(C=c)
        clf.fit(X, Y)
        score[c]=cross_val_score(clf, X, Y, scoring="accuracy", cv=5)
    return score


def NuSVM(X, Y):
    score = {}
    cs = [x / 10.0 for x in range(1, 5, 1)]
    i = 0
    for c in cs:
        #nu is an upper bound on the fraction of training errors and a lower bound of the fraction of support vectors.
        clf = NuSVC(nu=c, decision_function_shape='ovo')
        clf.fit(X, Y)
        score[i]=cross_val_score(clf, X, Y, scoring="accuracy", cv=5)
        i+=1
    return score

def run_all_models(X_train, Y_train, X_test, Y_test):

    scores = {}
    for m in model_list:
        scores[m].append(grid_search(X, Y, model_list[m], params_list[m], K))

    max_score = 0
    optimal_model = None
    optimal_params = None
    for model_name, score in scores.iteritmes():
        if np.max(score) > max_score:
            max_score = np.max(score)
            param_key, param_val = params_list[model_name]
            optimal_params = { param_key: param_val[np.argmax(score)]
            optimal_model = model_list[model_name]

    optimal_model.set_params(**optimal_params)
    optimal_model.fit(X_train, Y_train)
    Y_pred = optimal_model.predict(X_test)
    optimal_model.confusion_matrix(Y_test, Y_pred)


def printAll(X,Y):

    LR1_model = L1logisticReg(Xtrain, Ytrain)
    LR2_model = L2logisticReg(Xtrain, Ytrain)
    RF_model = randomForest(Xtrain, Ytrain)
    SVM_model = linearSVM(Xtrain, Ytrain)
    NuSVM_model = NuSVM(Xtrain, Ytrain)

    print("L1 Logistic Regression: ")
    print(LR1_model)
    print("L2 Logistic Regression: ")
    print(LR2_model)
    print("Random Forrest: ")
    print(RF_model)
    print("SVM: ")
    print(SVM_model)
    print("NuSVM: ")
    print(NuSVM_model)
