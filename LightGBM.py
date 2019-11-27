import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import sys
from xgb_dataset_generation import *

import lightgbm as lgb
import time

#train = base_expanded_df(alpha = 0.2, beta = 0.05, isValidation=True, save=True)
#test = base_expanded_df(alpha = 0.2, beta = 0.05, isValidation=False, save=True)
#train = pd.read_csv("dataset/expanded/base_expanded_train.csv")
#test = pd.read_csv("dataset/expanded/base_expanded_test.csv")

#train = adding_features(train, isValidation=True)
#test = adding_features(test, isValidation=False)
#train.to_csv('train_complete.csv', index=False)
#test.to_csv('test_complete.csv', index=False)

train = pd.read_csv("train_complete.csv")
test = pd.read_csv("test_complete.csv")

group = train.groupby('queried_record_id').size().values
ranker = lgb.LGBMRanker(device='gpu')
print('Start LGBM...')
t1 = time.time()
ranker.fit(train.drop(['queried_record_id', 'target', 'linked_id_idx'], axis=1), train['target'], group=group)
t2 = time.time()
print(f'Learning completed in {int(t2-t1)} seconds.')
predictions = ranker.predict(test.drop(['queried_record_id', 'linked_id_idx'], axis=1))
test['predictions'] = predictions
df_predictions = test[['queried_record_id', 'predicted_record_id', 'predictions']]

rec_pred = []
for (r,p) in zip(df_predictions.predicted_record_id, df_predictions.predictions):
    rec_pred.append((r, p))

df_predictions['rec_pred'] = rec_pred
df_predictions.to_csv('lgb_sub8_scores.csv', index = False)
group_queried = df_predictions[['queried_record_id', 'rec_pred']].groupby('queried_record_id').apply(lambda x: list(x['rec_pred']))
df_predictions = pd.DataFrame(group_queried).reset_index().rename(columns={0 : 'rec_pred'})
#rec_pred_corr = [[eval(x) for x in t] for t in tqdm(df_predictions.rec_pred)]
#df_predictions['rec_pred'] = rec_pred_corr

def reorder_preds(preds):
    sorted_list = []
    for i in range(len(preds)):
        l = sorted(preds[i], key=lambda t: t[1], reverse=True)
        l = [x[0] for x in l]
        sorted_list.append(l)
    return sorted_list

df_predictions['ordered_preds'] = reorder_preds(df_predictions.rec_pred.values)
df_predictions = df_predictions[['queried_record_id', 'ordered_preds']].rename(columns={'ordered_preds': 'predicted_record_id'})

new_col = []
for t in tqdm(df_predictions.predicted_record_id):
    new_col.append(' '.join([str(x) for x in t]))

df_predictions.predicted_record_id = new_col
df_predictions.to_csv('lgb_sub8.csv', index=False)
