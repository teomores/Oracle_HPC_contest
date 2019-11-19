import pandas as pd
from scipy import *
from scipy.sparse import *
from tqdm.auto import tqdm

"""
For each linked_id, computes:
- the number of null fields
- the percentage of non-nulls over the number of times that the
    linked_id appears
"""

df_train = pd.read_csv('../dataset/original/train.csv', escapechar="\\")
df_train = df_train.sort_values(by=['record_id']).reset_index(drop=True)
df_train.phone = df_train.phone.astype(str)
# checks if the phone is missing, in that case adds 1 to the dict for the linked_id
null_phone = {}
null_phone = dict(zip(list(set(df_train.linked_id.tolist())), [0 for x in range(len(df_train.linked_id.tolist()))]))
for ind,lid,phone in tqdm(zip(df_train.index, df_train.linked_id, df_train.phone)):
    if phone == 'nan':
        null_phone[lid]+=1
# compute the popularity to extract the percentage of non null phones
pop = {}
pop = dict(zip(list(set(df_train.linked_id.tolist())), [0 for x in range(len(df_train.linked_id.tolist()))]))
for ind,lid in tqdm(zip(df_train.index, df_train.linked_id)):
    pop[lid] +=1
# create the feature dataframe
feature = pd.DataFrame()
feature['linked_id'] = null_phone.keys()
feature['null_phone'] = null_phone.values()
feature['popularity'] = pop.values()
# compute the percentage
perc_phone = []
for m, p in tqdm(zip(feature.null_phone, feature.popularity)):
    perc_phone.append(int(100-m/p*100))
# end :)
feature['perc_non_null_phone'] = perc_phone
feature = feature.drop(['popularity'], axis=1)
print(feature)
# feature.to_csv('number_of_non_null_phone.csv', index = False)
