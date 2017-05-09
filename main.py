import pickle
from model import *
# from PCA import *

# from logistic import *


def main():
    # Parameter settings
    K = 5
    use_spark = True
    all_roc = True

    # Load data
    with open("data/train.p", "rb") as f:
        X_train, Y_train = pickle.load(f)
    with open("data/test.p", "rb") as f:
        X_test, Y_test = pickle.load(f)

    # drop weight
    X_train = np.delete(X_train, [i for i in range(19,29)], axis=1)
    X_test = np.delete(X_test, [i for i in range(19,29)], axis=1)

    if use_spark:
        from pyspark import SparkConf, SparkContext
        conf = SparkConf().setAppName("Readmission")
        conf = conf.setMaster("local[4]")
        sc = SparkContext(conf=conf)
    else:
        sc = None

    # Cross-Validation
    print ('Start training models with', K, '-fold cross validation...')
    plotScore = run_all_models(sc, X_train, Y_train, X_test, Y_test, K, all_roc)
    # run_a_model(X_train, Y_train, X_test, Y_test)
    # boxPlot(plotScore)

if __name__ == "__main__":
    main()
