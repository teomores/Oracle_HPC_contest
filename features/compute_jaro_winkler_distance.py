import pandas as pd
import numpy as np
import jaro
from tqdm.auto import tqdm

# TODO Ci impiega un po' forse meglio scriverla in Cython
# TODO estenderla anche per la similarità tra mail o altro

def compute_jaro_distance(df_exp, validation=True, Winkler=False):
    if validation:
        df_test = pd.read_csv(
            "/Users/alessiorussointroito/Documents/GitHub/Oracle_HPC_contest/dataset/validation/test.csv",
            escapechar="\\")
        df_train = pd.read_csv(
            "/Users/alessiorussointroito/Documents/GitHub/Oracle_HPC_contest/dataset/validation/train.csv",
            escapechar="\\")
    else:
        df_test = pd.read_csv(
            "/Users/alessiorussointroito/Documents/GitHub/Oracle_HPC_contest/dataset/original/test.csv",
            escapechar="\\")
        df_train = pd.read_csv(
            "/Users/alessiorussointroito/Documents/GitHub/Oracle_HPC_contest/dataset/original/train.csv",
            escapechar="\\")

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

    def extract_jaro_distance(queried_name, predicted_name):
        res = np.empty(len(queried_name), dtype=float)
        for i in tqdm(range(len(queried_name))):
            try:
                # res[i] = distance.get_jaro_distance(queried_name[i], predicted_name[i], winkler=False, scaling=0.1)
                res[i] = jaro.jaro_metric(queried_name[i], predicted_name[i])
            except:
                print(i)
        return res

    def extract_jarowinkler_distance(queried_name, predicted_name):
        res = np.empty(len(queried_name), dtype=float)
        for i in tqdm(range(len(queried_name))):
            try:
                # res[i] = distance.get_jaro_distance(queried_name[i], predicted_name[i], winkler=True, scaling=0.1)
                res[i] = jaro.jaro_winkler_metric(queried_name[i], predicted_name[i])
            except:
                print(i)
        return res

    if Winkler:
        df_exp['jarowinkler_distance'] = extract_jarowinkler_distance(df_exp.queried_name.values,
                                                                      df_exp.predicted_name.values)
        return df_exp['jarowinkler_distance']
    else:
        df_exp['jaro_distance'] = extract_jaro_distance(df_exp.queried_name.values, df_exp.predicted_name.values)
        return df_exp['jaro_distance']