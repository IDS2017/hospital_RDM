import pickle
from sklearn.utils import shuffle

# 0. Load meta
with open('dicts/meta.p', 'rb') as f:
    meta = pickle.load(f)

def get_cat(col_name, c):
    # init vector
    cate_idx = meta['used_cols'][col_name]['cate_idx']
    c_vec = [0]*(len(cate_idx)+1)

    # update vector
    if c == '?': # is missing value
        c_vec[-1] = 1
    else:
        c_vec[cate_idx[c]] = 1

    return c_vec

def get_num(col_name, n):
    # init vector
    if meta['used_cols'][col_name]['missing_cnt']: # has missing value
        n_vec = [0]*2
    else:
        n_vec = [0]

    # update vector
    if n == '?': # is missing value
        n_vec[1] = 1
        n_vec[0] = meta['used_cols'][col_name]['mean']
    else:
        n_vec[0] = float(n)

    return n_vec

def feature():
    # 1. Generating X, y
    X = []
    y = []

    cols = []
    header = True

    with open('data/diabetic_data.csv', 'rb') as f:
        for line in f:
            token = line.split(',')
            if header:
                cols = token
                continue
            row = []
            for i in range(len(token)):
                col = cols[i]                               # get column name
                if col not in meta['used_cols']:
                    continue
                data_type = meta['used_cols'][col]['type']  # get column type
                if data_type == 'categorical':
                    row.extend(get_cat(col, token[i]))
                else if data_type == 'numerical':
                    row.extend(get_num(col, token[i]))
                else if data_type == 'target':
                    y.append(meta['used_cols'][col]['class_idx'][token[i]])

    # 2. Shuffle
    X, y = shuffle(X, y, random_state=0)

    # 3. Split data into train / test set
    # data[0]: X_train, y_train
    # data[1]: X_test, y_test

    N = int(0.8*meta['total_instances'])
    data = [ [X[:N,:], y[:N,:]], [X[N:,:], y[:N,:]] ]

    with open("data/data.p") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    feature()
