import pandas as pd

df_train = pd.read_csv("dataset/original/train.csv", escapechar="\\")
df_test = pd.read_csv("dataset/original/test.csv", escapechar="\\")
df_train = df_train.sort_values(by='record_id')
df_test = df_test.sort_values(by='record_id')
df_test['target'] = df_test.record_id.str.split('-')
df_test['target'] = df_test.target.apply(lambda x: x[0])

df_test[['record_id', 'target']].to_csv("dataset/original/label_test.csv", index=False)