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

## Preprocess into x/y test/train
# get NN inputs and outputs
x, y = [], []
for i in range(len(eeg_eyes)):

    # convert list from dtype to float
    eeg_eyes.append(line.strip().split(','))

    # delete dc offset 4100uV from sensors
    x.append(np.array(eeg_eyes[i][0:-1]) - 4100)
    y.append(np.array(eeg_eyes[i][-1]))

# split data into test/train
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,
                                                    test_size=0.2)
# convert to np arrays
y_train = np.array(y_train)
y_test  = np.array(y_test)

# normalise test/train
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test  = tf.keras.utils.normalize(x_test,  axis=1)

## Create & Fit Model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape = x_train[0].shape))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))

STEPS_PER_EPOCH=100
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
      0.001,
      decay_steps=STEPS_PER_EPOCH*1000,
      decay_rate=1,
      staircase=False
)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train,y_train, epochs=40, steps_per_epoch=STEPS_PER_EPOCH,
          validation_data=(x_test, y_test))

## Test Model data
model.save('eeg.model')
new_model = tf.keras.models.load_model('eeg.model')
##
predictions = new_model.predict(x_test)
count = 0
for i in range(len(predictions)):
    print(np.argmax(predictions[i]))
    if (np.argmax(predictions[i]) == 0):
        count = count + 1

print(count*100/len(predictions))
