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
        score[c]  = []
        # Parameter c specifies regularisation strength, the smaller the stronger strength
        clf = linear_model.LogisticRegression(C = c, solver='liblinear')
        clf.fit(X, Y)
        #clf.predict(Xtest)
        score[c].append(cross_val_score(clf, X, Y, scoring="accuracy", cv=5))
    return score


def L2logisticReg(X,Y):
    score = {}
    #Regularisation strength, the smaller the stronger regularisation
    cs = [10**i for i in range(-5,5)]
    for c in cs:
        score[c]  = []
        # Parameter c specifies regularisation strength, the smaller the stronger strength
        clf = linear_model.LogisticRegression(C = c, solver='lbfgs')
        clf.fit(X, Y)
        #clf.predict(Xtest)
        score[c].append(cross_val_score(clf, X, Y, scoring="accuracy", cv=5))
    return score


def randomForest(X, Y):
    score = {}
    cs = [10**i for i in range(1,5)]
    for c in cs:
        score[c]  = []
        #n_estimators is The number of trees in the forest.
        clf = RandomForestClassifier(n_estimators=c)
        clf.fit(X, Y)
        #clf.predict(Xtest)
        score[c].append(cross_val_score(clf, X, Y, scoring="accuracy", cv=5))
    return score


def linearSVM(X, Y):
    score = {}
    cs = [10**i for i in range(1,5)]
    for c in cs:
        score[c] = []
        #Penalty parameter C of the error term
        clf = LinearSVC(C=c)
        clf.fit(X, Y)
        #clf.predict(Xtest)
        #score[c] = clf.score(Xtest, Ytest)
        score[c].append(cross_val_score(clf, X, Y, scoring="accuracy", cv=5))
    return score


def NuSVM(X, Y):
    score = {}
    cs = [x / 10.0 for x in range(1, 5, 1)]
    i=0
    for c in cs:
        score[i] = []
        #nu is an upper bound on the fraction of training errors and a lower bound of the fraction of support vectors.
        clf = NuSVC(nu=c, decision_function_shape='ovo')
        clf.fit(X, Y)
        #clf.predict(Xtest)
        #score[c] = clf.score(Xtest, Ytest)
        score[i].append(cross_val_score(clf, X, Y, scoring="accuracy", cv=5))
        i+=1
    return score
