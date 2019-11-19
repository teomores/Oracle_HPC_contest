import pandas as pd
from scipy import *
from scipy.sparse import *

"""
For each test record_id computes the length of the name.
"""

df_test = pd.read_csv('../dataset/original/test.csv', escapechar="\\")
df_test = df_test.sort_values(by=['record_id']).reset_index(drop=True)
df_test.name = df_test.name.astype(str)
col = []
for x in df_test.name:
    col.append(len(x))
feature = df_test[['record_id']]
feature['test_name_length'] = col
print(feature)
# feature.to_csv('test_name_length.csv', index=False)
