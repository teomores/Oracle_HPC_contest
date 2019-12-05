import pandas as pd
from tqdm import tqdm

"""
This features finds typos related to upper, title or lower case.
E.g:
    APPLE ---> Ok
    APPkE ---> Typo!
    Apple ---> Ok
    AppLE ---> Typo!
"""

def case_typo(isValidation):
    if isValidation:
        df_test = pd.read_csv('../dataset/validation/test.csv', escapechar="\\")
    else:
        df_test = pd.read_csv('../dataset/original/test.csv', escapechar="\\")
    df_test = df_test.sort_values(by=['record_id']).reset_index(drop=True)
    feature = df_test[['record_id','name']]
    feature.name = feature.name.astype(str)
    feature['case_typo'] = [check_string_words(s) for s in tqdm(list(feature.name))]
    final_feature = feature[['record_id','case_typo']]
    print(final_feature)
    if isValidation:
        file_path = '../dataset/validation/feature/case_typo.csv'
    else:
        file_path = '../dataset/original/feature/case_typo.csv'
    if os.path.exists(file_path):
        os.remove(file_path)
    final_feature.to_csv(filePafile_pathth, index=False)

def check_string_words(string):
    string = string.replace('-',' ')
    str_strip = string.split(' ')
    errors = []
    for s in str_strip:
        if s.islower() or s.isupper() or s.istitle():
            pass
        else:
            return 1
    return 0

case_typo(True)
case_typo(False)
