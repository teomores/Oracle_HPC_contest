from tqdm import tqdm
from scipy import *
from scipy.sparse import *
import pandas as pd
from utils import get_sub, threshold_matrix

s1 = load_npz('jaccard_uncleaned_name_300k_original.npz')
s2 = load_npz('jaccard_uncleaned_email_300k_original.npz')
s3 = load_npz('jaccard_uncleaned_phone_300k_original.npz')
s4 = load_npz('jaccard_uncleaned_address_300k_original.npz')
# massimo con s1+0.2*s2+0.2*s3+0.2*s4
sss = s1+0.2*s2+0.2*s3+0.2*s4

df_train = pd.read_csv("dataset/original/train.csv", escapechar="\\")
df_test = pd.read_csv("dataset/original/test.csv", escapechar="\\")
# ALWAYS sort the data by record_id
df_train = df_train.sort_values(by=['record_id']).reset_index(drop=True)
df_test = df_test.sort_values(by=['record_id']).reset_index(drop=True)
ss = get_sub(sss, df_train, df_test, 'prova_ngrams_con_tmail')
