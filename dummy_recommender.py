import pandas as pd
import string
import unidecode
from tqdm import tqdm
from collections import Counter
from scipy import *
from scipy.sparse import *
import similaripy as sim
from sys import argv
import time

"""
This recommender simply computes the cosine similarity between the names and
recommends the one ID that has the highest score.
TEST
        abcd...     linked1 linked2 ...
record1          X a
record2            b

Usage:
- python dummy_recommender.py save ---> to recompute the similarity
- python dummy_recommender.py load ---> to load the previously computed similarity

"""

def clean_names(df):
    df.name = df.name.astype(str) # convert to string
    df.name = df.name.str.lower() # lowercase
    df.name = df.name.str.translate(str.maketrans('', '', string.punctuation)) # remove punctuation
    # remove accented letters
    no_accents = []
    for s in df.name:
        no_accents.append(unidecode.unidecode(s))
    df.name = no_accents
    return df

def create_name_letters_matrix(df):
    df = df[['record_id','name']]
    df = clean_names(df)
    # create return matrix
    columns = ['record_id','name','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    name_letters_matrix = pd.DataFrame(columns=columns)
    name_letters_matrix.record_id = df.record_id.copy()
    name_letters_matrix.name = df.name.copy()
    # count occurence of each letter and add the columns to the return df
    for l in tqdm(['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']):
        new_col = []
        for (i,n) in zip(name_letters_matrix.index, name_letters_matrix.name):
            new_col.append(n.count(l))
        name_letters_matrix[l] = new_col
    return name_letters_matrix

if len(argv)==2:
    df_train = pd.read_csv("dataset/original/train.csv", escapechar="\\")
    df_test = pd.read_csv("dataset/original/test.csv", escapechar="\\")
    # extract referent for each group
    df_train = clean_names(df_train)
    group = df_train[['name', 'linked_id']].groupby('linked_id').apply(lambda x: list(x['name']))
    link_mc_name = {}
    for (l, names) in tqdm(zip(group.keys(), group)):
        link_mc_name[l] = Counter(names).most_common(1)[0][0]
    most_common_name = pd.DataFrame.from_dict(link_mc_name, orient='index', columns=['most_common_name'])
    df_train_clean = pd.merge(df_train, most_common_name, how='left', left_on='linked_id', right_index=True)
    df_train_clean = df_train_clean.drop_duplicates(subset=['linked_id','most_common_name']).drop(['record_id', 'name'], axis=1).reset_index(drop=True)
    df_train_clean = df_train_clean.rename(columns={"linked_id":"record_id", "most_common_name":"name"})
    m_train = create_name_letters_matrix(df_train_clean)
    m_test = create_name_letters_matrix(df_test)
    # now compute cosine similarity
    m_train_csr = csr_matrix(m_train.drop(['record_id','name'], axis=1))
    m_test_csr = csr_matrix(m_test.drop(['record_id','name'], axis=1))
    # save if passing argument "save"
    if argv[1]== "save":
        output = sim.cosine(m_test_csr, m_train_csr.T, k=10)
        save_npz(f'similarity_cosine_{time.time()}.npz', output.tocsr())
        out_csr = output.tocsr()
    elif argv[1] == "load":
        out_csr = load_npz('similarity_cosine.npz')
    sub = df_test[['record_id','name']]
    new_col = []
    for i in tqdm(range(out_csr.shape[0])):
        new_col.append(m_train.at[out_csr[i].argmax(), "record_id"])
    sub['predicted_record_id'] = new_col
    sub = sub.rename(columns={"record_id":"queried_record_id"})
    sub = sub[['queried_record_id','predicted_record_id']]
    sub.to_csv('sub_dummy.csv', index=False)
else:
    print("Usage:")
    print("python dummy_recommender.py save ---> to recompute the similarity")
    print("python dummy_recommender.py load ---> to load the previously computed similarity")
