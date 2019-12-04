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

import sys
import argparse
sys.path.append("..")
from utils import convert_phones, thresold_matrix

def ngrams(string, n=3):
    # TODO: change for numbers?
    string = string.encode("ascii", errors="ignore").decode() #remove non ascii chars
    string = string.lower() #make lower case
    string = string.translate(str.maketrans('', '', string_lib.punctuation)) # remove punctuation
    chars_to_remove = [")","(",".","|","[","]","{","}","'"]
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
    string = re.sub(rx, '', string) #remove the list of chars defined above
    string = string.replace('&', ' ')
    string = string.replace(',', ' ')
    string = string.replace('-', ' ')
    #string = string.title() # normalise case - capital at start of each word
    string = re.sub(' +',' ',string).strip() # get rid of multiple spaces and replace with a single space
    string = ' '+ string +' ' # pad names for ngrams...
    string = re.sub(r'[,-./]',r'', string)
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
# convert phones
df_test = convert_phones(df_test)
df_train = convert_phones(df_train)
# mi serve una colonna con tutti i telefoni su cui fare tfidf
all_phones = list(df_train.phone) + list(df_test.phone)
# daje con tfidf
vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
tf_idf_matrix = vectorizer.fit_transform(all_phones)
# split
tf_idf_train = tf_idf_matrix[:df_train.shape[0],:] # 691440 è la lunghezza del train
tf_idf_test = tf_idf_matrix[df_train.shape[0]:,:]
k = 2000
jac = sim.jaccard(tf_idf_test, tf_idf_train.T, k=k)
jac_csr = jac.tocsr()
m = thresold_matrix(jac_csr)
save_npz(f'jaccard_tfidf_phone_{args.split}_{k}_thresholded_09.npz', m)
