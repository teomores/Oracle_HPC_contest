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

import pandas as pd
import similaripy as sim
from scipy import *
from scipy.sparse import *
from tqdm.auto import tqdm
import numpy as np

import re
import string as string_lib
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
def ngrams(string, n=3):
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
    string = re.sub(r'[,-./]|\sBD',r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]

# first load the data
df_train = pd.read_csv("dataset/original/train.csv", escapechar="\\")
df_test = pd.read_csv("dataset/original/test.csv", escapechar="\\")
# ALWAYS sort the data by record_id
df_train = df_train.sort_values(by=['record_id']).reset_index(drop=True)
df_test = df_test.sort_values(by=['record_id']).reset_index(drop=True)
df_train.name = df_train.name.astype(str)
df_test.name = df_test.name.astype(str)
# mi serve una colonna con tutti i nomi su cui fare tfidf
all_names = list(df_train.name) + list(df_test.name)
# daje con tfidf
vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
tf_idf_matrix = vectorizer.fit_transform(all_names)
# split
tf_idf_train = tf_idf_matrix[:691440,:] # 691440 è la lunghezza del train
tf_idf_test = tf_idf_matrix[691440:,:]
cos_tfidf = sim.cosine(tf_idf_test, tf_idf_train.T, k=300)
save_npz('cos_tfidf_test_train_300.npz', cos_tfidf.tocsr())
