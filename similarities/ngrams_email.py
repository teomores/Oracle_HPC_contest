import pandas as pd
from scipy import *
from scipy.sparse import *
import similaripy as sim
from sklearn.feature_extraction.text import CountVectorizer
import argparse
import re

def remove_spaces(s, n=3):
    s = re.sub(' +',' ',s).strip()
    ngrams = zip(*[s[i:] for i in range(n)])
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
df_train.email = df_train.email.fillna('').astype(str)
df_test.email = df_test.email.fillna('').astype(str)
corpus = list(df_train.email) + list(df_test.email)
vectorizer = CountVectorizer(preprocessor = remove_spaces, analyzer=remove_spaces)
X = vectorizer.fit_transform(corpus)
X_train = X[:df_train.shape[0],:]
X_test = X[df_train.shape[0]:,:]
cosmatrixxx = sim.jaccard(X_test, X_train.T, k=300)
save_npz(f'jaccard_uncleaned_email_300k_{args.split}.npz', cosmatrixxx.tocsr())
