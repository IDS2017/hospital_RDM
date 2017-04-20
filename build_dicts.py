import pickle as pk
import pandas as pd

# 1. read the original data
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
    for unique_val in diabetic_data[col_name].unique():
        if 'diag_' not in col_name:
            meta['used_cols'][col_name]['categories'].append(unique_val)
            meta['used_cols'][col_name]['cate_cnt'][unique_val] = diabetic_data[col_name].value_counts()[unique_val]

# 4. get information from Numerical columns
for col_name in num_data.columns:
    meta['used_cols'][col_name] = dict(diabetic_data[col_name].describe())

# 5. set information whether the column value is numeric or categorical
for col_name in diabetic_data.columns:
    if col_name in num_data.columns:
        meta['used_cols'][col_name]['data_type'] = 'numeric'
    else:
        meta['used_cols'][col_name]['data_type'] = 'categorical'


print(meta)
    #print(col_name, ':', unique)
    #if col_name.contains('id') |
    #if col_type_dict.has_key('Categorical'):
    #    col_type_dict['Categorical']


# finally, write meta to a file
with open('dicts/data.pickle', 'wb') as f:
    pk.dump(meta, f, pk.HIGHEST_PROTOCOL)
