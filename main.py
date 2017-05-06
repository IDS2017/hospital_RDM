from feature import feature
from model import *

from pyspark import SparkConf, SparkContext
# from spark_sklearn.util import createLocalSparkSession


def main():
    # Parameter settings
    K = 5

    # Load data
    X_train, Y_train, X_test, Y_test = feature()

    conf = SparkConf().setAppName("Readmission")
    conf = conf.setMaster("local[4]")
    sc = SparkContext(conf=conf)

    # Cross-Validation
    print ('Start training models with', K, '-fold cross validation...')
    plotScore = run_all_models(sc, X_train, Y_train, X_test, Y_test, K)
    # boxPlot(plotScore)


if __name__ == "__main__":
    main()
