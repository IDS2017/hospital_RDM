import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC, LinearSVC
from sklearn.svm import NuSVC
from sklearn import linear_model, datasets
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from matplotlib.lines import Line2D



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
