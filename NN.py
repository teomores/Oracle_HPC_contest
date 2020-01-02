from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import logging
import os
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


# print("Loading the first validation...")
# val1_exp = pd.read_csv("dataset/validation/train_complete.csv")
# print("Loading the second validation...")
# val2_exp = pd.read_csv("dataset/validation_2/train_complete.csv")
# print("Loading the third validation...")
# val3_exp = pd.read_csv("dataset/validation_3/train_complete.csv")
# print('Concatenatin...')
# train_complete = pd.concat([val1_exp, val2_exp, val3_exp])

train_complete = pd.read_csv("train_complete_nn.csv")


y = train_complete['target']

X = train_complete.drop(['target'], axis=1).iloc[:,3:]
print('Building model')
# QUESTI FANNO 54923
# model = Sequential()
# model.add(Dense(32, input_dim=22, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
#
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# model.fit(X, y, validation_split=0.3, epochs=100, batch_size=300, callbacks=[es])
METRICS = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'),
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
]

model = Sequential()
model.add(Dense(32, input_dim=22, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.5))

output_bias = tf.keras.initializers.Constant(0.03)
model.add(Dense(1, activation='sigmoid',  bias_initializer=output_bias))

weight_for_0 = (1 / 3)*(100)/2.0
weight_for_1 = (1 / 97)*(100)/2.0

class_weight = {0: weight_for_0, 1: weight_for_1}

model.compile(loss='binary_crossentropy',
                optimizer=tf.keras.optimizers.Adam(lr=1e-3),
                metrics=METRICS,
                #mode='max',
                )

print('Fitting')
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(X, y, validation_split=0.3, epochs=100, batch_size=1000, callbacks=[es], class_weight=class_weight)

test_complete = pd.read_csv('dataset/original/test_complete.csv')

test_data = test_complete.drop(['linked_id_idx'], axis=1).iloc[:,3:]

pred = model.predict(test_data, verbose=1)

rounded = [x[0] for x in tqdm(pred)]
test_complete['predictions'] = rounded
test_complete[['queried_record_id','predicted_record_id','predicted_record_id_record','linked_id_idx','predictions']]
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
if os.path.exists('scores_nn_prova.csv'):
    os.remove('scores_nn_prova.csv')
df_predictions.to_csv('scores_nn_prova.csv', index=False)
