from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import logging
import os
from features.target import target

logging.basicConfig(level=logging.DEBUG)

# Set GPU memory growth
# Allows to only as much GPU memory as needed
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


logging.debug("Loading the first validation...")
val1_exp = pd.read_csv("dataset/validation/expanded/base_expanded_train.csv")
val1_exp['target'] = target(val1_exp, path = "dataset/validation")
logging.debug("Loading the second validation...")
val2_exp = pd.read_csv("dataset/validation_2/expanded/base_expanded_train.csv")
val2_exp['target'] = target(val2_exp, path = "dataset/validation_2")
logging.debug("Loading the third validation...")
val3_exp = pd.read_csv("dataset/validation_3/expanded/base_expanded_train.csv")
val3_exp['target'] = target(val3_exp, path = "dataset/validation_3")

train_complete = pd.concat([val1_exp, val2_exp, val3_exp])

train_complete = train_complete.drop(['linked_id_idx'], axis=1)

y = train_complete['target']

X = train_complete.drop(['target'], axis=1).iloc[:,3:]
print(X.columns)
model = Sequential()
model.add(Dense(32, input_dim=5, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(X, y, validation_split=0.3, epochs=100, batch_size=1000, callbacks=[es])

test_complete = pd.read_csv('dataset/expanded/base_expanded_test.csv')

test_data = test_complete.drop(['linked_id_idx'], axis=1).iloc[:,3:]

pred = model.predict(test_data, verbose=1)

rounded = [x[0] for x in tqdm(pred)]
test_complete['predictions'] = rounded
df_predictions = test_complete[['queried_record_id', 'predicted_record_id', 'predicted_record_id_record', 'predictions']]
rec_pred = []
for (l,p,record_id) in tqdm(zip(df_predictions.predicted_record_id, df_predictions.predictions, df_predictions.predicted_record_id_record)):
    rec_pred.append((l, p, record_id))

df_predictions['rec_pred'] = rec_pred
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
if os.path.exists('scores_nn_simple.csv'):
    os.remove('scores_nn_simple.csv')
df_predictions.to_csv('scores_nn_simple.csv', index=False)
