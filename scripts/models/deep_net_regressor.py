import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import trange
from tqdm import tqdm

import tensorflow as tf

train_csv = pd.read_csv("../../extracted_data/train.csv")
test_csv = pd.read_csv("../../extracted_data/test.csv")

# Get the list of categorical and continuous columns
catCols = ['cat'+f'{i}' for i in range(0, 10)]
conCols =  ['cont'+f'{i}' for i in range(0, 14)]

def encodeCategoricalColumns(train, test):
    for col in tqdm(catCols):
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col])
        test[col] = le.transform(test[col])
encodeCategoricalColumns(train_csv, test_csv)

train_csv = train_csv.sample(frac=1).reset_index(drop=True)
data = train_csv.drop(['id', 'target'], axis=1)
target = train_csv[['target']]

print(f"Data Shape: {data.shape}, Target shape: {target.shape}")

# Split the data
trainX, validX, trainY, validY = train_test_split(data, target, test_size=0.2)

print(trainX.shape, trainY.shape)
print(validX.shape, validY.shape)

def init_fc_model():
    inps = tf.keras.Input(shape=(None, trainX.shape[1]), batch_size=32)
    x = tf.keras.layers.Dense(256, 'selu')(inps)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, 'selu')(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Dense(64, 'tanh')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(32, 'tanh')(x)
    out = tf.keras.layers.Dense(1)(x)

    return tf.keras.Model(inputs=[inps], outputs=[out])

model = init_fc_model()
model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
model.summary()

model.fit(
    trainX,
    trainY,
    validation_data=(validX, validY),
    epochs=10
)

# Predictions for the baseline
preds = model.predict(test_csv.drop(['id'], axis=1).values)
submission['target'] = preds
submission.to_csv("submission.csv", index=None)