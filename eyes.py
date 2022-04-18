## Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split

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

    x.append(np.array(eeg_eyes[i][0:-1]))
    y.append(np.array(eeg_eyes[i][-1]))

# split data into test/train
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)

# normalise test/train
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test  = tf.keras.utils.normalize(x_test,  axis=1)

## Create NN-Model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))

