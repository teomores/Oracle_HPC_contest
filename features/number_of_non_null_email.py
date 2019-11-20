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

def number_of_non_null_email(isValidation):
    if isValidation:
        df_train = pd.read_csv('../dataset/validation/train.csv', escapechar="\\")
    else:
        df_train = pd.read_csv('../dataset/original/train.csv', escapechar="\\")

    df_train = df_train.sort_values(by=['record_id']).reset_index(drop=True)
    df_train.email = df_train.email.astype(str)
    # checks if the email is missing, in that case adds 1 to the dict for the linked_id
    null_mail = {}
    null_mail = dict(zip(list(set(df_train.linked_id.tolist())), [0 for x in range(len(list(set(df_train.linked_id.tolist()))))]))
    for lid,mail in tqdm(zip(df_train.linked_id, df_train.email)):
        if mail == 'nan':
            null_mail[lid]+=1
    # compute the popularity to extract the percentage of non null emails
    pop = {}
    pop = dict(zip(list(set(df_train.linked_id.tolist())), [0 for x in range(len(list(set(df_train.linked_id.tolist()))))]))
    for lid in tqdm(zip(df_train.linked_id)):
        pop[lid] +=1
    # create the feature dataframe
    feature = pd.DataFrame()
    feature['linked_id'] = null_mail.keys()
    feature['null_email'] = null_mail.values()
    feature['popularity'] = pop.values()
    # compute the percentage
    perc_mail = []
    for m, p in tqdm(zip(feature.null_email, feature.popularity)):
        perc_mail.append(int(100-m/p*100))
    # end :)
    feature['perc_non_null_email'] = perc_mail
    feature = feature.drop(['popularity'], axis=1)
    print(feature)
    if isValidation:
        feature.to_csv('../dataset/validation/feature/number_of_non_null_email.csv', index = False)
    else:
        feature.to_csv('../dataset/original/feature/number_of_non_null_email.csv', index = False)

#number_of_non_null_email(True)
number_of_non_null_email(False)
