import pandas as pd
from sklearn.model_selection import train_test_split

train = pd.read_csv("dataset/original/train.csv", escapechar="\\")

X_train, X_val, y_train, y_val = train_test_split(train, train['linked_id'] , test_size=0.33, random_state=42)
X_train.to_csv('dataset/validation/train.csv', index=False)
X_val.to_csv('dataset/validation/test.csv', index=False)
#y_train.to_csv('dataset/validation/y_train.csv', index=True)
#y_val.to_csv('dataset/validation/y_val.csv', index=True)


# Reading dataset keeping the saved indices:
# train = pd.read_csv('dataset/validation/X_train.csv', index_col=0)
