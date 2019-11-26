import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import sys
import xgboost as xgb

from xgb_dataset_generation import *


train = base_expanded_df(alpha = 0.3, beta = 0.2, isValidation=True, save=False)
test = base_expanded_df(alpha = 0.3, beta = 0.2, isValidation=False, save=False)

train = adding_features(train, isValidation=True)
test = adding_features(test, isValidation=False)

group = train.groupby('queried_record_id').size().values
ranker = xgb.XGBRanker()
ranker.fit(train.drop(['queried_record_id', 'target', 'linked_id_idx', 'linked_id'], axis=1), train['target'], group=group)

predictions = ranker.predict(test.drop(['queried_record_id', 'linked_id_idx'], axis=1))
test['predictions'] = predictions
df_predictions = test[['queried_record_id', 'predicted_record_id', 'predictions']]

rec_pred = []
for (r,p) in zip(df_predictions.predicted_record_id, df_predictions.predictions):
    rec_pred.append((r, p))

df_predictions['rec_pred'] = rec_pred
group_queried = df_predictions[['queried_record_id', 'rec_pred']].groupby('queried_record_id').apply(lambda x: list(x['rec_pred']))
df_predictions = pd.DataFrame(group_queried).reset_index().rename(columns={0 : 'rec_pred'})


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
df_predictions.to_csv('xgb_sub7.csv', index=False)