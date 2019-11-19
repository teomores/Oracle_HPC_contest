import pandas as pd
from tqdm import tqdm

"""
This feature computes the number of times that a linked_id appears in the
training set.
"""
df_train = pd.read_csv('../dataset/original/train.csv', escapechar="\\")
# costruisco un dizionario con chiavi i linked_id e valori tutti zeri
# poi ci sommo 1 ogni volta che compare il linked_id
# (con list.count(x) era lento...)
pop = {}
pop = dict(zip(list(set(df_train.linked_id.tolist())), [0 for x in range(len(df_train.linked_id.tolist()))]))
for ind,lid in tqdm(zip(df_train.index, df_train.linked_id)):
    pop[lid] +=1
feature = pd.DataFrame()
feature['linked_id'] = pop.keys()
feature['popularity'] = pop.values()
print(feature)
# feature.to_csv('popularity.csv', index = False)
