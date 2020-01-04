import pandas as pd
from scipy import *
from scipy.sparse import *
from tqdm.auto import tqdm
import os

"""
For each linked_id, computes:
- the number of null fields
- the percentage of non-nulls over the number of times that the
    linked_id appears
"""

def number_of_non_null_address(isValidation, path=""):
    if isValidation:
        train_path = os.path.join(path,'train.csv')
        df_train = pd.read_csv(train_path, escapechar="\\")
        #df_train = pd.read_csv('../dataset/validation/train.csv', escapechar="\\")
    else:
        df_train = pd.read_csv('../dataset/original/train.csv', escapechar="\\")

    df_train = df_train.sort_values(by=['record_id']).reset_index(drop=True)
    df_train.address = df_train.address.astype(str)
    # checks if the email is missing, in that case adds 1 to the dict for the linked_id
    null_address = {}
    null_address = dict(zip(list(set(df_train.linked_id.tolist())), [0 for x in range(len(list(set(df_train.linked_id.tolist()))))]))
    for lid,address in tqdm(zip(df_train.linked_id, df_train.address)):
        if address == 'nan':
            null_address[lid]+=1
    # compute the popularity to extract the percentage of non null emails
    pop = {}
    pop = dict(zip(list(set(df_train.linked_id.tolist())), [0 for x in range(len(list(set(df_train.linked_id.tolist()))))]))
    for lid in tqdm(df_train.linked_id):
        pop[lid] +=1
    # create the feature dataframe
    feature = pd.DataFrame()
    feature['linked_id'] = null_address.keys()
    feature['null_address'] = null_address.values()
    feature['popularity'] = pop.values()
    # compute the percentage
    perc_address = []
    for m, p in tqdm(zip(feature.null_address, feature.popularity)):
        perc_address.append(int(100-m/p*100))
    # end :)
    feature['perc_non_null_address'] = perc_address
    feature = feature.drop(['popularity'], axis=1)
    print(feature)

    if isValidation:
        feat_path = os.path.join(path, 'feature/number_of_non_null_address.csv')
        feature.to_csv(feat_path, index = False)
    else:
        feature.to_csv('../dataset/original/feature/number_of_non_null_address.csv', index = False)


number_of_non_null_address(True, path="../dataset/validation")
number_of_non_null_address(True, path="../dataset/validation_2")
number_of_non_null_address(True, path="../dataset/validation_3")
number_of_non_null_address(False)
