import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import py_stringmatching as sm

# TODO Ci impiega un po' forse meglio scriverla in Cython
# TODO estenderla anche per la similarità tra mail o altro

def compute_jaro_distance(df_exp, validation=True, Winkler=True, columns=['name', 'address', 'phone', 'email']):
    if validation:
        df_test = pd.read_csv("dataset/validation/test.csv", escapechar="\\")
        df_train = pd.read_csv("dataset/validation/train.csv",escapechar="\\")
    else:
        df_test = pd.read_csv("dataset/original/test.csv", escapechar="\\")
        df_train = pd.read_csv("dataset/original/train.csv", escapechar="\\")

    df_train = df_train.sort_values(by='record_id').reset_index(drop=True)
    df_test = df_test.sort_values(by='record_id').reset_index(drop=True)

    for c in columns:
        df_train[c] = df_train[c].fillna('')
        df_test[c] = df_test[c].fillna('')
        df_train[c] = df_train[c].astype(str)
        df_test[c] = df_test[c].astype(str)

    df_exp = df_exp.merge(df_test[['record_id'] + columns], how='left', left_on='queried_record_id',
                          right_on='record_id').drop('record_id', axis=1)
    #df_exp = df_exp.rename(columns={'name': 'queried_name'})

    df_exp = df_exp.merge(df_train[['record_id'] + columns], how='left', left_on='linked_id_idx', right_index=True, suffixes=('_queried','_predicted')).drop(
        'record_id', axis=1)
    #df_exp = df_exp.rename(columns={'name': 'predicted_name'})

    #print(f'NaN on queried_name: {df_exp.queried_name.isna().sum()}')
    #print(f'Nan on predicted_name: {df_exp.predicted_name.isna().sum()}')

    def extract_jaro_distance(queried_name, predicted_name):
        jw = sm.Jaro()
        res = np.empty(len(queried_name), dtype=float)
        for i in tqdm(range(len(queried_name))):
            try:
                # res[i] = distance.get_jaro_distance(queried_name[i], predicted_name[i], winkler=False, scaling=0.1)
                # res[i] = jaro.jaro_metric(queried_name[i], predicted_name[i])
                res[i] = jw.get_raw_score(queried_name[i], predicted_name[i])
            except:
                print(i)
        return res

    def extract_jarowinkler_distance(queried_name, predicted_name):
        jw = sm.JaroWinkler()
        res = np.empty(len(queried_name), dtype=float)
        for i in tqdm(range(len(queried_name))):
            try:
                # res[i] = distance.get_jaro_distance(queried_name[i], predicted_name[i], winkler=True, scaling=0.1)
                # res[i] = jaro.jaro_winkler_metric(queried_name[i], predicted_name[i])
                res[i] = jw.get_raw_score(queried_name[i], predicted_name[i])
            except:
                print(i)
        return res

    new_col_name = []

    for c in columns:
        #df_exp[c + '_queried'] = df_exp[c + '_queried'].fillna('')
        #df_exp[c + '_predicted'] = df_exp[c + '_predicted'].fillna('')

        df_exp[c + '_queried'] = df_exp[c + '_queried'].str.lower()
        df_exp[c + '_predicted'] = df_exp[c + '_predicted'].str.lower()

        if Winkler:
            df_exp['jw_' + c ] = extract_jarowinkler_distance(df_exp[c + '_queried'].values,
                                                                          df_exp[c + '_predicted'].values)
            new_col_name.append('jw_' + c )

        else:
            df_exp['j_' + c] = extract_jaro_distance(df_exp[c + '_queried'].values,
                                                             df_exp[c + '_predicted'].values)
            new_col_name.append('j_' + c )

    return df_exp[new_col_name]


