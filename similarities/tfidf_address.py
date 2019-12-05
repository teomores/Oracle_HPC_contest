import pandas as pd
import similaripy as sim
from scipy import *
from scipy.sparse import *
from tqdm import tqdm
import numpy as np

import re
import string as string_lib
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

import argparse
import sys
sys.path.append("..")
from utils import convert_phones

def ngrams(string, n=3):
    string = string.encode("ascii", errors="ignore").decode() #remove non ascii chars
    string = string.lower() #make lower case
    string = re.sub(' +',' ',string).strip() # get rid of multiple spaces and replace with a single space
    string = ' '+ string +' ' # pad names for ngrams...
    string = re.sub(r'[,-/]',r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]

#setup parser
parser = argparse.ArgumentParser()
parser.add_argument("-s","--split",
                    help="The dataset split to use",
                    choices=['original','validation'],
                    type=str,
                    required=True)
args = parser.parse_args()
# first load the data
df_train = pd.read_csv(f"../dataset/{args.split}/train.csv", escapechar="\\")
df_test = pd.read_csv(f"../dataset/{args.split}/test.csv", escapechar="\\")
# ALWAYS sort the data by record_id
df_train = df_train.sort_values(by=['record_id']).reset_index(drop=True)
df_test = df_test.sort_values(by=['record_id']).reset_index(drop=True)
df_train.address = df_train.address.astype(str)
df_test.address = df_test.address.astype(str)
# mi serve una colonna con tutti i telefoni su cui fare tfidf
all_adds = list(df_train.address) + list(df_test.address)
# daje con tfidf
vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
tf_idf_matrix = vectorizer.fit_transform(all_adds)
# split
tf_idf_train = tf_idf_matrix[:df_train.shape[0],:] # 691440 è la lunghezza del train
tf_idf_test = tf_idf_matrix[df_train.shape[0]:,:]
jac = sim.jaccard(tf_idf_test, tf_idf_train.T, k=300)
save_npz(f'jaccard_tfidf_address_{args.split}.npz', jac.tocsr())
