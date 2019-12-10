from tqdm import tqdm
from scipy import *
from scipy.sparse import *
import pandas as pd
from utils import get_sub, threshold_matrix
import numpy as np

s1 = load_npz('jaccard_tfidf_name_original.npz')
s2 = load_npz('jaccard_tfidf_email_original.npz')
s3 = load_npz('jaccard_tfidf_phone_original.npz')
s4 = load_npz('jaccard_tfidf_address_original.npz')
s5 = load_npz('similarity_cosine_complete_name.npz')
s6 = load_npz('similarity_cosine_complete_email.npz')

# massimo con name+0.05*email+0.2*phone+0.1*address
sss = 0.5 * (s1 + 0.4 * s5)+0.05*(s2 + 0.5* s6)+0.2*s3+0.1*s4

df_train = pd.read_csv("dataset/original/train.csv", escapechar="\\")
df_test = pd.read_csv("dataset/original/test.csv", escapechar="\\")
# ALWAYS sort the data by record_id
df_train = df_train.sort_values(by=['record_id']).reset_index(drop=True)
df_test = df_test.sort_values(by=['record_id']).reset_index(drop=True)
ss = get_sub(sss, df_train, df_test, 'hybrid_plus_cosine_name_email')
