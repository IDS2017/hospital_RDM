import pickle
import csv
import numpy as np
from build_dicts import get_ICD
from sklearn.utils import shuffle

# 0. Load meta
with open('dicts/meta.p', 'rb') as f:
    meta = pickle.load(f)

mask = ['max_glu_serum', 'A1Cresult']
con_cat = { 'max_glu_serum': ['Norm', '>200', '>300'],  \
            'A1Cresult': ['Norm', '>7', '>8'],  \
            'age': ['[0-10)','[10-20)','[20-30)','[30-40)','[40-50)','[50-60)','[60-70)','[70-80)','[80-90)','[90-100)']
}

def get_cat(col_name, c):
    if 'diag_' in col_name:
        c_vec_1 = [0]*20
        c_vec_2 = [0]*2
        c_vec_3 = [0]*3
        c_vec_4 = [0]*2
        c_vec_5 = [0]*10

        if meta['used_cols'][col_name]['missing_cnt']:
            c_vec_6 = [0]

        if c != '?':
            '''
            c_vec_1   index           : 0-19
            c_vec_2   is_diabete      : 0 (False), 1 (True)
            c_vec_3   which_type      : 0 (type1), 1 (type2), 2(not specified)
            c_vec_4   is_controlled   : 0 (False), 1 (True)
            c_vec_5   icd_code_detail : 0-9
            '''
            index, is_diabete, which_type, is_controlled, icd_code_detail = get_ICD(c)
            c_vec_1[index] = 1
            if is_diabete:
                c_vec_2[is_diabete] = 1
                c_vec_3[which_type] = 1
                c_vec_4[is_controlled] = 1
                c_vec_5[icd_code_detail] = 1
            else:
                c_vec_2[is_diabete] = 1
        else:
            c_vec_6[0] = 1

        if meta['used_cols'][col_name]['missing_cnt']:
            c_vec = c_vec_1 + c_vec_2 + c_vec_3 + c_vec_4 + c_vec_5 + c_vec_6
        else:
            c_vec = c_vec_1 + c_vec_2 + c_vec_3 + c_vec_4 + c_vec_5

    elif col_name in mask:
        # init vector
        cate_idx = meta['used_cols'][col_name]['cate_idx']
        if meta['used_cols'][col_name]['missing_cnt']:
            c_vec = [0]*(len(cate_idx)+2)  # mask + missing
        else:
            c_vec = [0]*(len(cate_idx)+1)  # mask

        # update vector
        if c != '?':
            if c == 'None':
                c_vec[-1] = 1
            else:
                for val in con_cat[col_name]:
                    if c != val:
                        c_vec[cate_idx[val]] = 1
        else:
            c_vec[-2] = 1

    else:
        # init vector
        cate_idx = meta['used_cols'][col_name]['cate_idx']
        if meta['used_cols'][col_name]['missing_cnt']:
            c_vec = [0]*(len(cate_idx)+1)  # missing
        else:
            c_vec = [0]*len(cate_idx)

        # update vector
        if c != '?':
            if c in cate_idx:
                if col_name in con_cat:
                    for val in con_cat[col_name]:
                        if c != val:
                            c_vec[cate_idx[val]] = 1
                else:
                    c_vec[cate_idx[c]] = 1
        else:
            c_vec[-1] = 1

    return c_vec


def get_num(col_name, n):
    # init vector
    if meta['used_cols'][col_name]['missing_cnt']:  # has missing value
        n_vec = [0]*2
    else:
        n_vec = [0]

    # update vector
    if n == '?':  # is missing value
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
    with open('data/diabetic_data.csv', 'rU') as f:
        diabetic_data = csv.reader(f, delimiter=',')
        for token in diabetic_data:
            if header:
                cols = token
                header = False
                continue
            row = []
            for i in range(len(token)):
                col = cols[i]                                    # get column name
                if col not in meta['used_cols']:
                    continue
                data_type = meta['used_cols'][col]['data_type']  # get column type
                if data_type == 'categorical':
                    row.extend(get_cat(col, token[i]))
                elif data_type == 'numeric':
                    row.extend(get_num(col, token[i]))
                elif data_type == 'target':
                    t = meta['used_cols'][col]['cate_idx'][token[i]]
                    if t < 2:  # NO, >30
                        y.append(0)
                    else:      # < 30
                        y.append(1)
            X.append(row)


    # 2. Shuffle
    X = np.array(X)
    y = np.array(y)
    X, y = shuffle(X, y, random_state=0)
    print (X.shape)
    print (y.shape)

    # 3. Split data into train / test set
    # train.p : X_train, y_train
    # test.p  : X_test, y_test

    N = int(0.8*meta['total_instances'])

    # return X[:N, :], y[:N], X[N:, :], y[N:]
    with open("data/train.p", "wb") as f:
        pickle.dump((X[:N, :], y[:N]), f)
    with open("data/test.p", "wb") as f:
        pickle.dump((X[N:, :], y[N:]), f)


if __name__ == "__main__":
    feature()
