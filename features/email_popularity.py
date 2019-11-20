import pandas as pd
from scipy import *
from scipy.sparse import *
from tqdm.auto import tqdm

"""
For each test email assigns the overall email popularity.
"""
def email_popularity(isValidation):
    if isValidation:
        df_train = pd.read_csv('../dataset/validation/train.csv', escapechar="\\")
        df_test = pd.read_csv('../dataset/validation/test.csv', escapechar="\\")
    else:
        df_train = pd.read_csv('../dataset/original/train.csv', escapechar="\\")
        df_test = pd.read_csv('../dataset/original/test.csv', escapechar="\\")

    df_train = df_train.sort_values(by=['record_id']).reset_index(drop=True)
    df_test = df_test.sort_values(by=['record_id']).reset_index(drop=True)
    # concat dataframes to compute the overall popularity
    full = df_train.email.append(df_test.email).reset_index(drop=True)
    full = [str(x).lower() for x in full if str(x)!="nan"]
    emails = dict(zip(list(set(full)), [0 for x in range(len(full))]))
    for email in tqdm(full):
        emails[email]+=1
    # feature creation
    feature = pd.DataFrame()
    feature['email'] = emails.keys()
    feature['email_popularity'] = emails.values()
    # now merge with test
    df_test.email = df_test.email.astype(str)
    df_test.email = df_test.email.str.lower()
    final_feature = df_test[['record_id', 'email']]
    final_feature = pd.merge(final_feature, feature, how='left', on=['email']).fillna(0)
    final_feature = final_feature[['record_id','email_popularity']]
    final_feature.email_popularity = final_feature.email_popularity.astype(int)
    print(final_feature)

    if isValidation:
        final_feature.to_csv('../dataset/validation/feature/email_popularity.csv', index=False)
    else:
        final_feature.to_csv('../dataset/original/feature/email_popularity.csv', index=False)




email_popularity(True)
email_popularity(False)