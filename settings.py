from sklearn.svm import LinearSVC
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier

model_list = {
    "Logistic_Regression": linear_model.LogisticRegression(solver='newton-cg', C=1.01, penalty='l2', class_weight='balanced'),
    "Random_Forest": RandomForestClassifier(class_weight='balanced'),
    "LinearSVM": LinearSVC(class_weight='balanced'),
    # "L1_L2_Logistic_Regression": linear_model.LogisticRegression(solver='liblinear', class_weight='balanced'),
    # "L2_Logistic_Regression": linear_model.LogisticRegression(penalty='l2', class_weight='balanced'),
}

params_list = {
    "Logistic_Regression": {'max_iter': [100, 500, 1000], 'C': [10**i for i in range(-3,1)]},
    "Random_Forest": {'n_estimators': [50, 100], \
                      'min_samples_split': [10*i for i in range(1,10)], \
                      'max_features': ['auto', 'log2'], \
                      'criterion': ['gini', 'entropy']},
    "LinearSVM": {'C': [10**i for i in range(-5,5)]},
    # "L1_L2_Logistic_Regression": {'penalty': ['l1', 'l2'], 'C': [0.01]},
    # "L2_Logistic_Regression": {'solver': ['newton-cg', 'lbfgs', 'sag'], 'C':[0.01]},
}
