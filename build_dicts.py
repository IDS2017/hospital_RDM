import pickle as pk
import pandas as pd

# 1. read the original data & split into training and target data
diabetic_data = pd.read_csv('data/diabetic_data.csv').drop('encounter_id', 1)
target_data = pd.read_csv('data/diabetic_data.csv')['readmitted']
meta = {'total_instances':diabetic_data.shape[0], 'used_cols':{}}

# 2. device Numerical and Categorical columns
num_data = diabetic_data._get_numeric_data()
for col_name in num_data:
    if 'id' in col_name:
        num_data = num_data.drop(col_name, 1)
cate_data = diabetic_data.drop(num_data.columns, 1)

# 3. get information from Categorical columns
for col_name in cate_data.columns:
    meta['used_cols'][col_name] = dict(diabetic_data[col_name].describe())
    meta['used_cols'][col_name]['categories'] = []
    meta['used_cols'][col_name]['cate_cnt'] = {}
    meta['used_cols'][col_name]['cate_idx'] = {}
    idx = 0
    for unique_val in diabetic_data[col_name].unique():
        if 'diag_' not in col_name:
            meta['used_cols'][col_name]['categories'].append(unique_val)
            meta['used_cols'][col_name]['cate_cnt'][unique_val] = diabetic_data[col_name].value_counts()[unique_val]
            meta['used_cols'][col_name]['cate_idx'][unique_val] = idx
            idx = idx + 1

# 4. get information from Numerical columns
for col_name in num_data.columns:
    meta['used_cols'][col_name] = dict(diabetic_data[col_name].describe())

# 5. set information whether the column value is numeric or categorical
for col_name in diabetic_data.columns:
    cnt_table = diabetic_data[col_name].value_counts()
    if '?' in cnt_table.index:
        meta['used_cols'][col_name]['missing_cnt'] = cnt_table['?']
    if col_name in num_data.columns:
        meta['used_cols'][col_name]['data_type'] = 'numeric'
    else:
        meta['used_cols'][col_name]['data_type'] = 'categorical'

# 6. set information for last column 'target'
#meta['used_cols']['readmitted']['data_type'] = 'target'
#idx = 0
#for unique_val in target_data.index[0].unique():
#    meta['used_cols'][col_name] = {}
#    meta['used_cols'][col_name]['cate_idx'] = {}
#    meta['used_cols'][col_name]['cate_idx'][unique_val] = idx
#    idx = idx + 1

print(meta)

# 7. finally, write meta to a file
with open('dicts/data.pickle', 'wb') as f:
    pk.dump(meta, f, pk.HIGHEST_PROTOCOL)
