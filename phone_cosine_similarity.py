import pandas as pd
import similaripy as sim
from scipy import *
from scipy.sparse import *
from tqdm import tqdm
import string
import unidecode

def create_phone_numbers_matrix(df):
    df = df[['record_id','phone']]
    df.phone =  df.phone.fillna('')
    df.phone = df.phone.astype(str) # convert to string
    col = []
    for x in df.phone:
        col.append(x.split('.')[0])
    df['phone'] = col
    # create return matrix
    columns = ['record_id','phone','0','1','2',
              '3','4','5','6','7','8','9']
    phone_numbers_matrix = pd.DataFrame(columns=columns)
    phone_numbers_matrix.record_id = df.record_id.copy()
    phone_numbers_matrix.phone = df.phone.copy()
    # count occurence of each letter and add the columns to the return df
    for l in tqdm(['0','1','2',
                   '3','4','5','6','7','8','9']):
        new_col = []
        for (i,n) in zip(phone_numbers_matrix.index, phone_numbers_matrix.phone):
            new_col.append(n.count(l))
        phone_numbers_matrix[l] = new_col
    return phone_numbers_matrix

# first load the data
df_train = pd.read_csv("dataset/original/train.csv", escapechar="\\")
df_test = pd.read_csv("dataset/original/test.csv", escapechar="\\")
# ALWAYS sort the data by record_id
df_train = df_train.sort_values(by=['record_id']).reset_index(drop=True)
df_test = df_test.sort_values(by=['record_id']).reset_index(drop=True)
# create matrices
m_train = create_phone_numbers_matrix(df_train)
m_test = create_phone_numbers_matrix(df_test)
m_train_csr = csr_matrix(m_train.drop(['record_id','phone'], axis=1))
m_test_csr = csr_matrix(m_test.drop(['record_id','phone'], axis=1))
# compute similarity
output = sim.cosine(m_test_csr, m_train_csr.T, k=300)
save_npz('similarity_cosine_complete_phone_300.npz', output.tocsr())
