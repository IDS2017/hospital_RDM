import pickle

def main():
    # Parameter settings
    K = 5

    # Initialization
    train_err = 0
    test_err = 0

    # Load data
    with open("data/data.p", "r") as f:
        data = pickle.load(f)

    X_train, y_train = data[0]
    X_test, y_test = data[1]

    # Cross-Validation
    print "Start training models with', K, '-fold cross validation..."



    # Build Model (with whole data)

    # Evaluation

if __name__ == "__main__":
    main()
