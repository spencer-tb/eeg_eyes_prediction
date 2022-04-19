## Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Set tensorflow compiler flag
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

## Load in EEG data
data_path = '/home/spencer-tb/Documents/eeg/1_eyes/EEG_Eyes.txt'
data_eeg_eyes = []
with open(data_path) as file:
    for line in file:
        data_eeg_eyes.append(line.strip().split(','))

# convert eeg data list from dtype to float
eeg_eyes = []
for i in range(len(data_eeg_eyes)):
    tmp = []
    for j in range(len(data_eeg_eyes[i])):
        tmp.append(float(data_eeg_eyes[i][j]))
        eeg_eyes.append(tmp)
# tensor format
eeg_eyes = np.array(eeg_eyes).astype(np.float32)

## Preprocess into x/y - test/train
# split data into test/train/validation
train_size = int(0.8 * len(eeg_eyes))
test_size  = int(0.1 * len(eeg_eyes))
val_size   = int(0.1 * len(eeg_eyes))
train_ds = eeg_eyes[0:train_size]
test_ds  = eeg_eyes[train_size:test_size+train_size]
val_ds   = eeg_eyes[train_size+test_size:val_size+test_size+train_size]

# split into x/y
def split_xy(ds, ds_size):
    x, y = [], []
    for i in range(ds_size):
        # delete dc offset 4100uV from sensors
        x.append(ds[i][0:-1] - 4100)
        y.append(ds[i][-1])
    return x, y

x_train, y_train = split_xy(train_ds, len(train_ds))
x_test, y_test = split_xy(test_ds, len(test_ds))
x_val, y_val = split_xy(val_ds, len(val_ds))

# normalise x_test/train/val
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test  = tf.keras.utils.normalize(x_test , axis=1)
x_val   = tf.keras.utils.normalize(x_val  , axis=1)

## Create & Fit Model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape = x_train[0].shape))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, np.array(y_train), epochs=40, steps_per_epoch=100,
          validation_data=(x_val, np.array(y_val)))

## Test Model
print("Evaluate on test data")
results = model.evaluate(x_test, np.array(y_test), batch_size=128)
print("test loss, test acc:", results)
