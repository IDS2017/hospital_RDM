import pickle
import numpy as np

def sample():
    with open("data/train.p", "rb") as f:
        X_train, Y_train = pickle.load(f)
    TARGET = 9028

    sample_X_train = np.zeros((TARGET*2,376))
    sample_Y_train = np.zeros(TARGET*2)

    M = 0
    N = 0
    D = 0

    for i in range(len(Y_train)):
        if Y_train[i] == 0:  # NO, >30
            if M < TARGET:
                sample_X_train[D,:] = X_train[i,:]
                sample_Y_train[D] = Y_train[i]
                M += 1
                D += 1
        else:                # <30
            if N < TARGET:
                sample_X_train[D,:] = X_train[i,:]
                sample_Y_train[D] = Y_train[i]
                N += 1
                D += 1
        if D == TARGET*2:
            break
    print (M, N, D)
    with open("data/sample_train.p", "wb") as f:
        pickle.dump((sample_X_train, sample_Y_train), f)

if __name__ == "__main__":
    sample()
