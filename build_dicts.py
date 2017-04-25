import pickle as pk
import pandas as pd


# 1. read the original data & split into training and target data
diabetic_data = pd.read_csv('data/diabetic_data.csv').drop('encounter_id', 1)
meta = {'total_instances': diabetic_data.shape[0], 'used_cols': {}}

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


#print(meta)

# 6. finally, write meta to a file
with open('dicts/meta.p', 'wb') as f:
    pk.dump(meta, f, pk.HIGHEST_PROTOCOL)



def get_ICD(icd_code):
    icd_code = icd_code.split('.')[0]
    # icd_code_index = {0:(1,139), 1:(140,239), 2:(240,279), 3:(280,289), 4:(290,319), 5:(320,359),
    #                  6:(360,389), 7:(390,459), 8:(460,519), 9:(520,579), 10:(580,629),
    #                  11:(630, 679), 12:(680, 709), 13:(710, 739), 14:(740, 759), 15:(760, 779),
    #                  16:(780, 799), 17:(800, 999), 18:('E', 'V')}
    icd_code_index = [('1', '139'), ('140', '239'), ('240', '279'), ('280', '289'), ('290', '319'),
                      ('320', '359'), ('360', '389'), ('390', '459'), ('460', '519'), ('520', '579'),
                      ('580', '629'), ('630', '679'), ('680', '709'), ('710', '739'), ('740', '759'),
                      ('760', '779'), ('780', '799'), ('800', '999'), ('E', 'V')]

    if icd_code_index[0][0] <= icd_code <= icd_code_index[0][1]:
        index = 0
    elif icd_code_index[1][0] <= icd_code <= icd_code_index[1][1]:
        index = 1
    elif icd_code_index[2][0] <= icd_code <= icd_code_index[2][1]:
        index = 2
    elif icd_code_index[3][0] <= icd_code <= icd_code_index[3][1]:
        index = 3
    elif icd_code_index[4][0] <= icd_code <= icd_code_index[4][1]:
        index = 4
    elif icd_code_index[5][0] <= icd_code <= icd_code_index[5][1]:
        index = 5
    elif icd_code_index[6][0] <= icd_code <= icd_code_index[6][1]:
        index = 6
    elif icd_code_index[7][0] <= icd_code <= icd_code_index[7][1]:
        index = 7
    elif icd_code_index[8][0] <= icd_code <= icd_code_index[8][1]:
        index = 8
    elif icd_code_index[9][0] <= icd_code <= icd_code_index[9][1]:
        index = 9
    elif icd_code_index[10][0] <= icd_code <= icd_code_index[10][1]:
        index = 10
    elif icd_code_index[11][0] <= icd_code <= icd_code_index[11][1]:
        index = 11
    elif icd_code_index[12][0] <= icd_code <= icd_code_index[12][1]:
        index = 12
    elif icd_code_index[13][0] <= icd_code <= icd_code_index[13][1]:
        index = 13
    elif icd_code_index[14][0] <= icd_code <= icd_code_index[14][1]:
        index = 14
    elif icd_code_index[15][0] <= icd_code <= icd_code_index[15][1]:
        index = 15
    elif icd_code_index[16][0] <= icd_code <= icd_code_index[16][1]:
        index = 16
    elif icd_code_index[17][0] <= icd_code <= icd_code_index[17][1]:
        index = 17
    elif icd_code_index[18][0] in icd_code:
        index = 18
    elif icd_code_index[18][1] in icd_code:
        index = 18
    else:
        index = 19

    return index

def print_icd_code_index():
    for value in diabetic_data.diag_1:
            print(value, ": ", get_ICD(value))

