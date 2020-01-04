from tqdm import tqdm
from scipy import *
from scipy.sparse import *
import pandas as pd
from utils import get_sub, threshold_matrix
import numpy as np
import os

s1 = load_npz('dataset/original/similarities/jaccard_uncleaned_name_300k_original_3ngrams.npz')
s2 = load_npz('dataset/original/similarities/jaccard_uncleaned_email_300k_original_2ngrams.npz')
s3 = load_npz('dataset/original/similarities/jaccard_uncleaned_phone_300k_original_2ngrams.npz')
s4 = load_npz('dataset/original/similarities/jaccard_uncleaned_address_300k_original_2ngrams.npz')
#s5 = load_npz('similarity_cosine_name_300.npz')

# massimo con name+0.05*email+0.2*phone+0.1*address
sss = s1 + 0.2 * s2 + 0.2 * s3 + 0.2 * s4

df_train = pd.read_csv("dataset/original/train.csv", escapechar="\\")
df_test = pd.read_csv("dataset/original/test.csv", escapechar="\\")
# ALWAYS sort the data by record_id
df_train = df_train.sort_values(by=['record_id']).reset_index(drop=True)
df_test = df_test.sort_values(by=['record_id']).reset_index(drop=True)
if os.path.exists('prova.csv'):
    os.remove('prova.csv')
ss = get_sub(sss, df_train, df_test, 'prova')
