import pandas as pd
from tqdm import tqdm
import os

"""
This feature computes the number of times that a linked_id appears in the
training set.
"""

def linked_id_popularity(isValidation, path=""):

    if isValidation:
        #df_train = pd.read_csv('../dataset/validation/train.csv', escapechar="\\")
        train_path = os.path.join(path,  'train.csv')
        df_train = pd.read_csv(train_path, escapechar="\\")
    else:
        df_train = pd.read_csv('../dataset/original/train.csv', escapechar="\\")

    # costruisco un dizionario con chiavi i linked_id e valori tutti zeri
    # poi ci sommo 1 ogni volta che compare il linked_id
    # (con list.count(x) era lento...)
    pop = {}
    pop = dict(zip(list(set(df_train.linked_id.tolist())), [0 for x in range(len(list(set(df_train.linked_id.tolist()))))]))
    for lid in tqdm(df_train.linked_id):
        pop[lid] +=1
    feature = pd.DataFrame()
    feature['linked_id'] = pop.keys()
    feature['popularity'] = pop.values()
    print(feature)
    if isValidation:
        feat_path = os.path.join(path, 'feature/linked_id_popularity.csv')
        feature.to_csv(feat_path, index = False)
    else:
        feature.to_csv('../dataset/original/feature/linked_id_popularity.csv', index = False)


linked_id_popularity(True, path="../dataset/validation")
linked_id_popularity(True, path="../dataset/validation_2")
linked_id_popularity(True, path="../dataset/validation_3")
linked_id_popularity(False)
