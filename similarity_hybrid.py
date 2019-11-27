from tqdm import tqdm
from scipy import *
from scipy.sparse import *
import pandas as pd
from utils import get_sub

s1 = load_npz('jaccard_tfidf_name_original.npz')
s4 = load_npz('similarity_jaccard_name_300.npz')
s3 = load_npz('similarity_cosine_complete_name.npz')

sss = s1+s4*0.8
df_train = pd.read_csv("dataset/original/train.csv", escapechar="\\")
df_test = pd.read_csv("dataset/original/test.csv", escapechar="\\")
# ALWAYS sort the data by record_id
df_train = df_train.sort_values(by=['record_id']).reset_index(drop=True)
df_test = df_test.sort_values(by=['record_id']).reset_index(drop=True)
ss = get_sub(sss, df_train, df_test, 'name_tfidf_jaccard08')
