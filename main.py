import pickle
from feature import feature
from model import run_all_models, boxPlot


def main():
    # Parameter settings
    K = 5

    # Initialization
    train_err = 0
    test_err = 0

    # Load data
    # with open("data/data.p", "r") as f:
        # data = pickle.load(f)
    X_train, Y_train, X_test, Y_test = feature()

    # X_train, y_train = data[0]
    # X_test, y_test = data[1]

    # Cross-Validation
    print "Start training models with', K, '-fold cross validation..."
    plotScore = run_all_models(X_train, Y_train, X_test, Y_test, K)
    boxPlot(plotScore)
    plt.show()


if __name__ == "__main__":
    main()
