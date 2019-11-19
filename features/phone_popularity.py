import pandas as pd
from scipy import *
from scipy.sparse import *
from tqdm.auto import tqdm

"""
For each test phone assigns the overall phone popularity.
"""

df_train = pd.read_csv('../dataset/original/train.csv', escapechar="\\")
df_train = df_train.sort_values(by=['record_id']).reset_index(drop=True)
df_test = pd.read_csv('../dataset/original/test.csv', escapechar="\\")
df_test = df_test.sort_values(by=['record_id']).reset_index(drop=True)
# concat dataframes to compute the overall popularity
full = df_train.phone.append(df_test.phone).reset_index(drop=True)
full = [str(x).lower() for x in full if str(x)!="nan"]
phones = dict(zip(list(set(full)), [0 for x in range(len(full))]))
for phone in tqdm(full):
    phones[phone]+=1
# feature creation
feature = pd.DataFrame()
feature['phone'] = phones.keys()
feature['phone_popularity'] = phones.values()
# now merge with test
df_test.phone = df_test.phone.astype(str)
df_test.phone = df_test.phone.str.lower()
final_feature = df_test[['record_id', 'phone']]
final_feature = pd.merge(final_feature, feature, how='left', on=['phone']).fillna(0)
final_feature = final_feature[['record_id','phone_popularity']]
final_feature.phone_popularity = final_feature.phone_popularity.astype(int)
print(final_feature)
# final_feature.to_csv('phone_popularity.csv', index=False)
