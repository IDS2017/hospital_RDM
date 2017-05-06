from feature import feature
from model import *

from spark_sklearn.util import createLocalSparkSession


def main():
    # Parameter settings
    K = 5

    # Initialization
    # train_err = 0
    # test_err = 0

    # Load data
    # with open("data/data.p", "r") as f:
        # data = pickle.load(f)
    X_train, Y_train, X_test, Y_test = feature()

    # X_train, y_train = data[0]
    # X_test, y_test = data[1]

    spark = createLocalSparkSession()
    sc = spark.sparkContext
    # Cross-Validation
    print ('Start training models with', K, '-fold cross validation...')
    plotScore = run_all_models(sc, X_train, Y_train, X_test, Y_test, K)
    boxPlot(plotScore)

    spark.stop()
    SparkSession._instantiatedContext = None


if __name__ == "__main__":
    main()
