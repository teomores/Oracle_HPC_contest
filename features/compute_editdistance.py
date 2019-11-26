import pandas as pd
import numpy as np
from tqdm import tqdm
import editdistance


def compute_editdistance(df_exp, validation=True):
    if validation:
        df_test = pd.read_csv("../dataset/validation/test.csv", escapechar="\\")
        df_train = pd.read_csv("../dataset/validation/train.csv", escapechar="\\")
    else:
        df_test = pd.read_csv("../dataset/original/test.csv", escapechar="\\")
        df_train = pd.read_csv("../dataset/original/train.csv", escapechar="\\")

    df_train = df_train.sort_values(by='record_id').reset_index(drop=True)
    df_test = df_test.sort_values(by='record_id').reset_index(drop=True)

    df_exp = df_exp.merge(df_test[['record_id', 'name']], how='left', left_on='queried_record_id',
                          right_on='record_id').drop('record_id', axis=1)
    df_exp = df_exp.rename(columns={'name': 'queried_name'})

    df_exp = df_exp.merge(df_train[['record_id', 'name']], how='left', left_on='linked_id_idx', right_index=True).drop(
        'record_id', axis=1)
    df_exp = df_exp.rename(columns={'name': 'predicted_name'})

    print(f'NaN on queried_name: {df_exp.queried_name.isna().sum()}')
    print(f'Nan on predicted_name: {df_exp.predicted_name.isna().sum()}')

    df_exp['queried_name'] = df_exp.queried_name.fillna('')
    df_exp['predicted_name'] = df_exp.predicted_name.fillna('')

    df_exp['queried_name'] = df_exp.queried_name.str.lower()
    df_exp['predicted_name'] = df_exp.predicted_name.str.lower()

    def extract_editdistance(queried_name, predicted_name):
        res = np.empty(len(queried_name), dtype=int)
        for i in tqdm(range(len(queried_name))):
            try:
                res[i] = editdistance.eval(queried_name[i], predicted_name[i])
            except:
                print(i)
        return res

    df_exp['editdistance'] = extract_editdistance(df_exp.queried_name.values, df_exp.predicted_name.values)
    df_exp = df_exp.drop(['queried_name', 'predicted_name'], axis=1)
    return df_exp['editdistance']


