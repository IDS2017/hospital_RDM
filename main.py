from feature import feature
from model import *

from spark_sklearn.util import createLocalSparkSession


def main():
    # Parameter settings
    K = 5

    # Load data
    X_train, Y_train, X_test, Y_test = feature()

    spark = createLocalSparkSession()
    sc = spark.sparkContext

    # Cross-Validation
    print ('Start training models with', K, '-fold cross validation...')
    plotScore = run_all_models(sc, X_train, Y_train, X_test, Y_test, K)
    # boxPlot(plotScore)

    spark.stop()
    SparkSession._instantiatedContext = None


if __name__ == "__main__":
    main()
