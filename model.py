from settings import *

from sklearn.svm import SVC, LinearSVC
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier

from spark_sklearn import GridSearchCV as SPGridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, roc_auc_score, cohen_kappa_score, make_scorer


def grid_search(sc, X, Y, m, cs, K):
    roc_auc_scorer = make_scorer(roc_auc_score)

    if sc is None:  # sklearn
        clf = GridSearchCV(m, cs, cv=K, scoring=roc_auc_scorer)
        clf.fit(X,Y)
        print (clf.cv_results_)
        return clf.cv_results_['mean_test_score']
    else:           # sklearn on spark
        clf = SPGridSearchCV(sc, m, cs, cv=K, scoring=roc_auc_scorer, n_jobs=-1)
        clf.fit(X,Y)
        grid_result = clf.grid_scores_
        param_score = []
        for i in range(len(grid_result)):
            param_score.append((grid_result[i][0], grid_result[i][1]))
        print (param_score)
        return param_score


def run_all_models(sc, X_train, Y_train, X_test, Y_test, K=5, all_roc=True):
    scores = {}
    for m in model_list:
        print ('run', m)
        scores[m] = grid_search(sc, X_train, Y_train, model_list[m], params_list[m], K)

    print (scores)
    if sc is None:  # sklearn
        pass
    else:           # sklearn on spark
        if all_roc:
            optimal_param = {}
            for model_name, param_score in scores.items():
                max_score = 0
                for param, score in param_score:
                    if score > max_score:
                        max_score = score
                        optimal_param[model_name] = param
            for m in optimal_param:
                print (m)
                optimal_model = model_list[m]
                optimal_params = optimal_param[m]

                # Build Model (with whole data)
                optimal_model.set_params(**optimal_params)
                print ('optimal_settings', optimal_model)
                optimal_model.fit(X_train, Y_train)
                Y_pred = optimal_model.predict(X_test)

                # Evaluation
                roc_auc = roc_auc_score(Y_test, Y_pred)
                print ('confusion_matrix', confusion_matrix(Y_test, Y_pred))
                print ('roc_auc_score', roc_auc)
                print ('cohen_kappa_score', cohen_kappa_score(Y_test, Y_pred))
                utils.dump(Y_test, "drop_Y_test_%s" % (m))
                utils.dump(Y_pred, "drop_Y_pred_%s" % (m))
        else:
            max_score = 0
            for model_name, param_score in scores.items():
               for param, score in param_score:
                   if score > max_score:
                       max_score = score
                       optimal_params = param
                       optimal_model = model_list[model_name]

            # Build Model (with whole data)
            optimal_model.set_params(**optimal_params)
            print ('optimal_settings', optimal_model)
            optimal_model.fit(X_train, Y_train)
            Y_pred = optimal_model.predict(X_test)

            # Evaluation
            print ('confusion_matrix', confusion_matrix(Y_test, Y_pred))
            print ('roc_auc_score', roc_auc_score(Y_test, Y_pred))
            print ('cohen_kappa_score', cohen_kappa_score(Y_test, Y_pred))

    return scores

def run_a_model(X_train, Y_train, X_test, Y_test):
    optimal_model = linear_model.LogisticRegression()
    optimal_params = {'class_weight': 'balanced', 'C': 0.01, 'solver': 'newton-cg'}

    # Build Model (with whole data)
    optimal_model.set_params(**optimal_params)
    print ('optimal_model', optimal_model)
    optimal_model.fit(X_train, Y_train)
    Y_pred = optimal_model.predict(X_test)
    # Evaluation
    print ('confusion_matrix', confusion_matrix(Y_test, Y_pred))
    print ('roc_auc_score', roc_auc_score(Y_test, Y_pred))
    print ('cohen_kappa_score', cohen_kappa_score(Y_test, Y_pred))

    optimal_params = {'class_weight': 'None', 'C': 0.01, 'solver': 'newton-cg'}
    # Build Model (with whole data)
    optimal_model.set_params(**optimal_params)
    print ('optimal_model', optimal_model)
    optimal_model.fit(X_train, Y_train)
    Y_pred = optimal_model.predict(X_test)
    # Evaluation
    print ('confusion_matrix', confusion_matrix(Y_test, Y_pred))
    print ('roc_auc_score', roc_auc_score(Y_test, Y_pred))
    print ('cohen_kappa_score', cohen_kappa_score(Y_test, Y_pred))
