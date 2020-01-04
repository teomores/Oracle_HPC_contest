import pandas as pd
import numpy as np
from tqdm import tqdm
import sys

from dataset.original.evaluation_script import *
from sklearn.metrics import average_precision_score

def expand_df(s):
    # Expand df
    new_df = []
    for (q, l, r, sc) in tqdm(zip(s.queried_record_id, s.ordered_linked, s.ordered_record, s.ordered_scores)):
        for x in range(len(l)):
            new_df.append((q, l[x], r[x], sc[x]))

    sub = pd.DataFrame(new_df, columns=['queried_record_id', 'predicted_linked_id', 'predicted_record_id', 'score'])
    return sub

def restricted_df(s):
    restricted_pred = []
    max_delta = 2.0
    for (q, sc, rec, l) in tqdm(zip(s.queried_record_id, s.ordered_scores, s.ordered_record, s.ordered_linked)):
        for x in range(len(sc)):
            if x == 0: # Il primo elemento predetto si mette sempre [quello con score più alto]
                restricted_pred.append((q, sc[x], rec[x]))
            else:
                if x >= 10:
                    continue
                elif (sc[0] - sc[x] < max_delta) or (l[0] == l[x]):   # se le predizioni hanno uno scores che dista < max_delta dalla prima allora si inseriscono
                    restricted_pred.append((q, sc[x], rec[x]))
                else:
                    continue
    restricted_df = pd.DataFrame(restricted_pred, columns=['queried_record_id', 'scores', 'predicted_record_id'])
    return restricted_df

def compute_f1_score(precision, recall):
    F1 = 2 * (precision * recall) / (precision + recall)
    return F1

# Get LightGBM predictions
score_path = "lgb_predictions_full.csv"
s = pd.read_csv(score_path)

s.ordered_scores = [eval(x) for x in s.ordered_scores]
s.ordered_linked = [eval(x) for x in s.ordered_linked]
s.ordered_record = [eval(x) for x in s.ordered_record]

sub = expand_df(s)

# Load train and test
train = pd.read_csv("dataset/original/train.csv", escapechar="\\")
test = pd.read_csv("dataset/original/test.csv", escapechar="\\")

# Add Linked_id to test queries
test['linked_id'] = test.record_id.str.split("-")
test['linked_id'] = test.linked_id.apply(lambda x: x[0])
test.linked_id = test.linked_id.astype(int)

# Top10 Submission
print("Top10 Submission:")
precision = precision_at_k(sub[['queried_record_id', 'predicted_record_id']], train.set_index("record_id"), test.set_index("record_id"))
print(f'Precision@10: {precision["AveragePrecision"]}')

recall = recall_at_k(sub[['queried_record_id', 'predicted_record_id']], train.set_index("record_id"), test.set_index("record_id"))
print(f'Recall@10: {recall["AveragePrecision"]}')

f1 = compute_f1_score(precision['AveragePrecision'], recall['AverageRecall'])
print(f'F1-score: {f1}')

# Restricted Submission
print("Restricted Submission")
restricted_df = restricted_df(s)

precision_rest = precision_at_k(restricted_df[['queried_record_id', 'predicted_record_id']], train.set_index("record_id"), test.set_index("record_id"))
print(f'Precision@10: {precision_rest["AveragePrecision"]}')


recall_rest = recall_at_k(restricted_df[['queried_record_id', 'predicted_record_id']], train.set_index("record_id"), test.set_index("record_id"))
print(f'Recall@10: {recall_rest["AveragePrecision"]}')

f1_rest = compute_f1_score(precision_rest['AveragePrecision'], recall_rest['AverageRecall'])
print(f'F1-score: {f1_rest}')




