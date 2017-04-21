import pickle as pk
import pandas as pd

# 1. read the original data & split into training and target data
diabetic_data = pd.read_csv('data/diabetic_data.csv').drop('encounter_id', 1)
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
        meta['used_cols'][col_name]['categories'].append(str(unique_val))
        meta['used_cols'][col_name]['cate_cnt'][str(unique_val)] = diabetic_data[col_name].value_counts()[unique_val]
        meta['used_cols'][col_name]['cate_idx'][str(unique_val)] = idx
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
    if col_name == 'readmitted':
        meta['used_cols'][col_name]['data_type'] = 'target'

print(meta)

# 6. finally, write meta to a file
with open('dicts/meta.p', 'wb') as f:
    pk.dump(meta, f, pk.HIGHEST_PROTOCOL)
