import pickle as pk
import pandas as pd

# 1. read the original data
diabetic_data = pd.read_csv('data/diabetic_data.csv').drop('encounter_id', 1)
meta = {'total_instances':diabetic_data.shape[0], 'categorical':{}, 'numerical':{}}

# 2. find Numerical columns
num_data = diabetic_data._get_numeric_data()
for col_name in num_data:
    if 'id' in col_name:
        num_data = num_data.drop(col_name, 1)

# 3. form a dictionary for Categorical and Numerical columns with column names
for col_name in diabetic_data.columns:
    if col_name in num_data.columns:
        meta['numerical'][col_name] = {}
    else:
        meta['categorical'][col_name] = {}

# 4. get information from Categorical columns
for col_name in meta['categorical']:
    meta['categorical'][col_name] = dict(diabetic_data[col_name].describe())
    meta['categorical'][col_name]['categories'] = []
    meta['categorical'][col_name]['cate_cnt'] = {}
    for unique_val in diabetic_data[col_name].unique():
        if 'diag_' not in col_name:
            meta['categorical'][col_name]['categories'].append(unique_val)
            meta['categorical'][col_name]['cate_cnt'][unique_val] = diabetic_data[col_name].value_counts()[unique_val]

# 5. get information from Numerical columns
for col_name in meta['numerical']:
    meta['numerical'][col_name] = dict(diabetic_data[col_name].describe())

print(meta)
    #print(col_name, ':', unique)
    #if col_name.contains('id') |
    #if col_type_dict.has_key('Categorical'):
    #    col_type_dict['Categorical']


# finally, write meta to a file
with open('dicts/data.pickle', 'wb') as f:
    pk.dump(meta, f, pk.HIGHEST_PROTOCOL)
