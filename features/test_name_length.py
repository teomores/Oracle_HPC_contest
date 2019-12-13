import pandas as pd
from scipy import *
from scipy.sparse import *
import os

"""
For each test record_id computes the length of the name.
"""
def test_name_length(isValidation, path=""):
    if isValidation:
        test_path = os.path.join(path, 'test.csv')
        df_test = pd.read_csv(test_path, escapechar="\\")
        #df_test = pd.read_csv('../dataset/validation/test.csv', escapechar="\\")
    else:
        df_test = pd.read_csv('../dataset/original/test.csv', escapechar="\\")

    df_test = df_test.sort_values(by=['record_id']).reset_index(drop=True)
    df_test.name = df_test.name.astype(str)
    col = []
    for x in df_test.name:
        col.append(len(x))
    feature = df_test[['record_id']]
    feature['test_name_length'] = col
    print(feature)

    if isValidation:
        feat_path = os.path.join(path, 'feature/test_name_length.csv')
        feature.to_csv(feat_path, index=False)
    else:
        feature.to_csv('../dataset/original/feature/test_name_length.csv', index=False)


test_name_length(True, path="../dataset/validation_2")
test_name_length(True, path="../dataset/validation_3")

#test_name_length(False)