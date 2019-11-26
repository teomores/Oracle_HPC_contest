from tqdm import tqdm
from scipy import *
from scipy.sparse import *
import pandas as pd
from utils import get_sub

s1 = load_npz('../jaccard_tfidf_test_train_300.npz')
s2 = load_npz('../jaccard_tfidf_phone.npz')
s3 = load_npz('../jaccard_tfidf_email.npz')
s = s1+0.1*s2+0.05*s3
df_train = pd.read_csv("../dataset/original/train.csv", escapechar="\\")
df_test = pd.read_csv("../dataset/original/test.csv", escapechar="\\")
# ALWAYS sort the data by record_id
df_train = df_train.sort_values(by=['record_id']).reset_index(drop=True)
df_test = df_test.sort_values(by=['record_id']).reset_index(drop=True)
ss = get_sub(s, df_train, df_test, 'name_01phone_005mail') 
