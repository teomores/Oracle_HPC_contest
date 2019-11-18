import pandas as pd
import similaripy as sim
from scipy import *
from scipy.sparse import *
from tqdm import tqdm
import string
import unidecode

def create_email_letters_matrix(df):
    df = df[['record_id','email']]
    df.email =  df.email.fillna('')
    df.email = df.email.astype(str) # convert to string
    df.email = df.email.str.lower() # lowercase
    no_accents = []
    for s in df.email:
        no_accents.append(unidecode.unidecode(s))
    df.email = no_accents
    # create return matrix
    columns = ['record_id','email','a','b','c','d','e','f','g','h','i','j','k','l',
               'm','n','o','p','q','r','s','t','u','v','w','x','y','z','0','1','2',
              '3','4','5','6','7','8','9']
    email_letters_matrix = pd.DataFrame(columns=columns)
    email_letters_matrix.record_id = df.record_id.copy()
    email_letters_matrix.email = df.email.copy()
    # count occurence of each letter and add the columns to the return df
    for l in tqdm(['a','b','c','d','e','f','g','h','i','j','k','l','m','n',
                   'o','p','q','r','s','t','u','v','w','x','y','z','0','1','2',
                   '3','4','5','6','7','8','9']):
        new_col = []
        for (i,n) in zip(email_letters_matrix.index, email_letters_matrix.email):
            new_col.append(n.count(l))
        email_letters_matrix[l] = new_col
    return email_letters_matrix

# first load the data
df_train = pd.read_csv("dataset/original/train.csv", escapechar="\\")
df_test = pd.read_csv("dataset/original/test.csv", escapechar="\\")
# ALWAYS sort the data by record_id
df_train = df_train.sort_values(by=['record_id']).reset_index(drop=True)
df_test = df_test.sort_values(by=['record_id']).reset_index(drop=True)
# create matrices
m_train = create_email_letters_matrix(df_train)
m_test = create_email_letters_matrix(df_test)
m_train_csr = csr_matrix(m_train.drop(['record_id','email'], axis=1))
m_test_csr = csr_matrix(m_test.drop(['record_id','email'], axis=1))
# compute similarity
output = sim.cosine(m_test_csr, m_train_csr.T, k=100)
save_npz('similarity_cosine_complete_email.npz', output.tocsr())
