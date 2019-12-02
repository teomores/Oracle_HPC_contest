import pandas as pd
import numpy as np

def target(df_exp_train):
    df_val = pd.read_csv("dataset/validation/test.csv", escapechar="\\")
    df_exp_train = df_exp_train.merge(df_val[['record_id', 'linked_id']], how='left', left_on='queried_record_id', right_on='record_id').drop('record_id', axis=1)

    def extract_target(predicted, linked):
        res = np.where(predicted == linked, 1, 0)
        return res

    df_exp_train['target'] = extract_target(df_exp_train.predicted_record_id.values, df_exp_train.linked_id.values)
    #return df_exp_train.drop(['linked_id'], axis=1)
    return df_exp_train['target']
