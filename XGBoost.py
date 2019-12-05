import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import sys
import xgboost as xgb

from xgb_dataset_generation import *


train = base_expanded_df(alpha = 0.2, beta = 0.05, isValidation=True, save=True)
test = base_expanded_df(alpha = 0.2, beta = 0.05, isValidation=False, save=True)
#train = pd.read_csv("dataset/expanded/base_expanded_train.csv")
#test = pd.read_csv("dataset/expanded/base_expanded_test.csv")

train = adding_features(train, isValidation=True)
test = adding_features(test, isValidation=False)
train.to_csv('dataset/expanded/train_complete.csv', index=False)
test.to_csv('dataset/expanded/test_complete.csv', index=False)

train = pd.read_csv("train_complete.csv")
test = pd.read_csv("test_complete.csv")

group = train.groupby('queried_record_id').size().values
ranker = xgb.XGBRanker()
ranker.fit(train.drop(['queried_record_id', 'target', 'linked_id_idx'], axis=1), train['target'], group=group)

predictions = ranker.predict(test.drop(['queried_record_id', 'linked_id_idx'], axis=1))
test['predictions'] = predictions
df_predictions = test[['queried_record_id', 'predicted_record_id', 'predicted_record_id_record', 'predictions']]

rec_pred = []
for (r,p,l,record_id) in zip(df_predictions.predicted_record_id, df_predictions.predictions, df_predictions.predicted_record_id_record):
    rec_pred.append((r, p, l, record_id))

df_predictions['rec_pred'] = rec_pred
df_predictions.to_csv('xgb_sub8_scores.csv', index = False)
group_queried = df_predictions[['queried_record_id', 'rec_pred']].groupby('queried_record_id').apply(lambda x: list(x['rec_pred']))
df_predictions = pd.DataFrame(group_queried).reset_index().rename(columns={0 : 'rec_pred'})

def reorder_preds(preds):
    ordered_lin = []
    ordered_score = []
    ordered_record = []
    for i in range(len(preds)):
        l = sorted(preds[i], key=lambda t: t[1], reverse=True)
        lin = [x[0] for x in l]
        s = [x[1] for x in l]
        r = [x[2] for x in l]
        ordered_lin.append(lin)
        ordered_score.append(s)
        ordered_record.append(r)
    return ordered_lin, ordered_score, ordered_record

df_predictions['ordered_linked'], df_predictions['ordered_scores'], df_predictions['ordered_record'] = reorder_preds(df_predictions.rec_pred.values)
#df_predictions = df_predictions[['queried_record_id', 'ordered_preds']].rename(columns={'ordered_preds': 'predicted_record_id'})

#new_col = []
#for t in tqdm(df_predictions.predicted_record_id):
#    new_col.append(' '.join([str(x) for x in t]))

#df_predictions.predicted_record_id = new_col
#df_predictions.to_csv('xgb_sub8.csv', index=False)
