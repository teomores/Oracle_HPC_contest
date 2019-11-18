import pandas as pd
from scipy import *
from scipy.sparse import *
from tqdm import tqdm

sim_name = load_npz('similarity_cosine_complete_name.npz')
sim_email = load_npz('similarity_cosine_complete_email.npz')
hybrid = sim_name # + 0.3 * sim_email
#
df_train = pd.read_csv('dataset/original/train.csv', escapechar="\\")
df_test = pd.read_csv('dataset/original/test.csv', escapechar="\\")
df_train = df_train.sort_values(by=['record_id']).reset_index(drop=True)
df_test = df_test.sort_values(by=['record_id']).reset_index(drop=True)
# recupera i linked id (senza dupl)
# TODO: attaccare gli indici del record_id con lo score maggiore al linked_id (per le features di xgboost)
linid = []
for x in tqdm(range(1,df_test.shape[0])):
    linid.append(df_train.iloc[hybrid[x].nonzero()[1][hybrid[x].data.argsort()[::-1]]].drop_duplicates(['linked_id'])['linked_id'].values[:10])
#
sub = pd.DataFrame()
sub['queried_record_id'] = df_test.record_id
sub['predicted_record_id'] = linid
strings = []
for t in tqdm(sub.predicted_record_id):
    strings.append(' '.join([str(x) for x in t]))
sub.to_csv('mimmo.csv', index=False)
