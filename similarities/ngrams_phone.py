import pandas as pd
from scipy import *
from scipy.sparse import *
import similaripy as sim
from sklearn.feature_extraction.text import CountVectorizer
import argparse
import re
import os

def remove_spaces(s, n=3):
    s = re.sub(' +',' ',s).strip()
    ngrams = zip(*[s[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]

def convert_phones(df_in):
    """
    This functions transforms the phone column from scientific notation to readable string
    format, e.g. 1.2933+E10 to 12933000000
    : param df_in : the original df with the phone in scientific notation
    : return : the clean df
    """
    df = df_in.copy()
    df.phone = df.phone.fillna('').astype(str)
    df.phone = [p.split('.')[0] for p in df.phone]
    return df

#setup parser
parser = argparse.ArgumentParser()
parser.add_argument("-s","--split",
                    help="The dataset split to use",
                    choices=['original','validation','validation_2','validation_3'],
                    type=str,
                    required=True)
parser.add_argument("-m","--mode",
                    help="The mode to run.",
                    choices=['thresh','knn'],
                    type=str,
                    required=True)
args = parser.parse_args()
# first load the data
df_train = pd.read_csv(f"../dataset/{args.split}/train.csv", escapechar="\\")
df_test = pd.read_csv(f"../dataset/{args.split}/test.csv", escapechar="\\")
# ALWAYS sort the data by record_id
df_train = df_train.sort_values(by=['record_id']).reset_index(drop=True)
df_test = df_test.sort_values(by=['record_id']).reset_index(drop=True)
# convert phones
df_test = convert_phones(df_test)
df_train = convert_phones(df_train)
corpus = list(df_train.phone) + list(df_test.phone)
vectorizer = CountVectorizer(preprocessor = remove_spaces, analyzer=remove_spaces)
X = vectorizer.fit_transform(corpus)
X_train = X[:df_train.shape[0],:]
X_test = X[df_train.shape[0]:,:]
if args.mode == 'thresh':
    cosmatrixxx = sim.jaccard(X_test, X_train.T, k=2000)
    cosmatrixxx.data[cosmatrixxx.data <= 0.25] = 0
else:
    cosmatrixxx = sim.jaccard(X_test, X_train.T, k=300)


if not os.path.isdir(f"../dataset/{args.split}/similarities"):
    os.makedirs(f"../dataset/{args.split}/similarities")

save_npz(f'../dataset/{args.split}/similarities/jaccard_uncleaned_phone_{args.split}_2ngrams_{args.mode}.npz', cosmatrixxx.tocsr())
