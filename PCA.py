from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model

from spark_sklearn import GridSearchCV as SPGridSearchCV
from sklearn.metrics import confusion_matrix, roc_auc_score, cohen_kappa_score, make_scorer

pipe = Pipeline([
    ('pca', PCA()),
    # ('rf', RandomForestClassifier(class_weight='balanced'))
    ('lr', linear_model.LogisticRegression(solver='lbfgs', class_weight='balanced'))  # L2-reg
])

param_grid = {
    'pca__n_components': [250, 300, 350],
    'lr__C': [0.01],
    # 'rf__n_estimators': [10, 50, 100],
    # 'rf__min_samples_split': [10*i for i in range(1,10)],
    # 'rf__max_features': ['auto', 'log2'],
    # 'rf__criterion': ['gini', 'entropy']
}

def grid_search(sc, X, Y, K):
    roc_auc_scorer = make_scorer(roc_auc_score)
    clf = SPGridSearchCV(sc, pipe, cv=K, n_jobs=2, param_grid=param_grid, scoring=roc_auc_scorer)
    clf.fit(X,Y)
    grid_result = clf.grid_scores_
    param_score = []
    for i in range(len(grid_result)):
        param_score.append((grid_result[i][0], grid_result[i][1]))

    print (param_score)
    return param_score

def run_all_models(sc, X_train, Y_train, X_test, Y_test, K=5):
    param_score = grid_search(sc, X_train, Y_train, K)

    max_score = 0
    for param, score in param_score:
        if score > max_score:
            max_score = score
            optimal_params = param
            optimal_model = linear_model.LogisticRegression(solver='lbfgs', class_weight='balanced')

    # Build Model (with whole data)
    optimal_model.set_params(**optimal_params)
    print ('optimal_model', optimal_model)
    optimal_model.fit(X_train, Y_train)
    Y_pred = optimal_model.predict(X_test)

    # Evaluation
    print ('confusion_matrix', confusion_matrix(Y_test, Y_pred))
    print ('roc_auc_score', roc_auc_score(Y_test, Y_pred))
    print ('cohen_kappa_score', cohen_kappa_score(Y_test, Y_pred))

    return scores
