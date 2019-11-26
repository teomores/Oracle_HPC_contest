import pandas as pd
import similaripy as sim
from scipy import *
from scipy.sparse import *
from tqdm import tqdm
import string
import unidecode
from sklearn.feature_extraction.text import TfidfTransformer

def create_name_letters_matrix(df):
    df = df[['record_id','name']]
    df.name = df.name.astype(str) # convert to string
    df.name = df.name.str.lower() # lowercase
    df.name = df.name.str.translate(str.maketrans('', '', string.punctuation)) # remove punctuation
    # remove accented letters
    no_accents = []
    for s in df.name:
        no_accents.append(unidecode.unidecode(s))
    df.name = no_accents
    # create return matrix
    columns = ['record_id','name','a','b','c','d','e','f','g','h','i','j','k','l',
               'm','n','o','p','q','r','s','t','u','v','w','x','y','z']
    name_letters_matrix = pd.DataFrame(columns=columns)
    name_letters_matrix.record_id = df.record_id.copy()
    name_letters_matrix.name = df.name.copy()
    # count occurence of each letter and add the columns to the return df
    for l in tqdm(['a','b','c','d','e','f','g','h','i','j','k','l','m','n',
                   'o','p','q','r','s','t','u','v','w','x','y','z']):
        new_col = []
        for (i,n) in zip(name_letters_matrix.index, name_letters_matrix.name):
            new_col.append(n.count(l))
        name_letters_matrix[l] = new_col
    return name_letters_matrix

# first load the data
df_train = pd.read_csv("../dataset/original/train.csv", escapechar="\\")
df_test = pd.read_csv("../dataset/original/test.csv", escapechar="\\")
# ALWAYS sort the data by record_id
df_train = df_train.sort_values(by=['record_id']).reset_index(drop=True)
df_test = df_test.sort_values(by=['record_id']).reset_index(drop=True)
# create matrices
m_train = create_name_letters_matrix(df_train)
m_test = create_name_letters_matrix(df_test)
m_train_csr = csr_matrix(m_train.drop(['record_id','name'], axis=1))
m_test_csr = csr_matrix(m_test.drop(['record_id','name'], axis=1))
# normalizzo
m_train_csr = TfidfTransformer().fit_transform(m_train_csr)
m_test_csr = TfidfTransformer().fit_transform(m_test_csr)
# compute similarity
output = sim.cosine(m_test_csr, m_train_csr.T, k=300)
save_npz('cos_name_300_tfidf.npz', output.tocsr())
