import pandas as pd
from tqdm import tqdm
import os

"""
This features finds typos related to upper, title or lower case in the address field
"""

def case_typo_address(isValidation):
    if isValidation:
        df_test = pd.read_csv('../dataset/validation/test.csv', escapechar="\\")
    else:
        df_test = pd.read_csv('../dataset/original/test.csv', escapechar="\\")
    df_test = df_test.sort_values(by=['record_id']).reset_index(drop=True)
    feature = df_test[['record_id','address']]
    feature.address = feature.address.fillna('').astype(str)
    feature['case_typo_address'] = [check_string_words(s) for s in tqdm(list(feature.address))]
    final_feature = feature[['record_id','case_typo_address']]
    print(final_feature)
    if isValidation:
        file_path = '../dataset/validation/feature/case_typo_address.csv'
    else:
        file_path = '../dataset/original/feature/case_typo_address.csv'
    if os.path.exists(file_path):
        os.remove(file_path)
    final_feature.to_csv(file_path, index=False)

def check_string_words(string):
    if string == '':
        return 0
    else:
        string = string.replace('-',' ')
        str_strip = string.split(' ')
        errors = []
        for s in str_strip:
            if s.islower() or s.isupper() or s.istitle():
                pass
            else:
                return 1
        return 0

case_typo_address(True)
case_typo_address(False)
