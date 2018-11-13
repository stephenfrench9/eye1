import csv
import os
import math
from keras.models import Sequential, model_from_json
from os import listdir
import keras
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import SGD
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from scipy.misc import imread
from scipy.misc import imshow
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import datetime

os.chdir("/ralston")
print(os.listdir("."))

# -----------------------Initialize train labels, label names----------------------------

train_labels = pd.read_csv("train.csv")
labels = {
    0:  "Nucleoplasm",
    1:  "Nuclear membrane",
    2:  "Nucleoli",
    3:  "Nucleoli fibrillar center",
    4:  "Nuclear speckles",
    5:  "Nuclear bodies",
    6:  "Endoplasmic reticulum",
    7:  "Golgi apparatus",
    8:  "Peroxisomes",
    9:  "Endosomes",
    10:  "Lysosomes",
    11:  "Intermediate filaments",
    12:  "Actin filaments",
    13:  "Focal adhesion sites",
    14:  "Microtubules",
    15:  "Microtubule ends",
    16:  "Cytokinetic bridge",
    17:  "Mitotic spindle",
    18:  "Microtubule organizing center",
    19:  "Centrosome",
    20:  "Lipid droplets",
    21:  "Plasma membrane",
    22:  "Cell junctions",
    23:  "Mitochondria",
    24:  "Aggresome",
    25:  "Cytosol",
    26:  "Cytoplasmic bodies",
    27:  "Rods & rings"
}

reverse_train_labels = dict((v,k) for k,v in labels.items())

def fill_targets(row):
    row.Target = np.array(row.Target.split(" ")).astype(np.int)
    for num in row.Target:
        name = labels[int(num)]
        row.loc[name] = 1
    return row

for key in labels.keys():
    train_labels[labels[key]] = 0

train_labels = train_labels.apply(fill_targets, axis=1)

# ---------------------- Read using keras.utils.Sequence --------------------------
base = "train/"
blue = "_blue.png"
red = "_red.png"
yellow = "_yellow.png"
green = "_green.png"

from skimage.io import imread
from skimage.transform import resize
import numpy as np
from keras.utils import Sequence


# Here, `x_set` is list of path to the images
# and `y_set` are the associated classes.

class CIFAR10Sequence(Sequence):

    def __init__(self, train_labels, batch_size):
        self.train_labels = train_labels
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.train_labels) / float(self.batch_size)))

    def __getitem__(self, idx):
        trials = len(self.train_labels)
        y = np.ones(self.batch_size)
        x = np.ones((1, 512, 512, 4))
        # x = np.ones((1, 512, 512))
        for i in range(self.batch_size):
            sample = i + idx * self.batch_size
            b = imread(base + train_labels.at[sample, 'Id'] + red).reshape((512, 512, 1))
            b = b / np.std(b)
            b = b - b.mean()
            r = imread(base + train_labels.at[sample, 'Id'] + blue).reshape((512, 512, 1))
            r = r / np.std(r)
            r = r - r.mean()
            ye = imread(base + train_labels.at[sample, 'Id'] + yellow).reshape((512, 512, 1))
            ye = ye / np.std(ye)
            ye = ye - ye.mean()
            g = imread(base + train_labels.at[sample, 'Id'] + green).reshape((512, 512, 1))
            # g = imread(base + train_labels.at[sample, 'Id'] + green).reshape((1, 512, 512))
            g = g / np.std(g)
            g = g - g.mean()
            im = np.append(b, r, axis=2)
            im = np.append(im, ye, axis=2)
            im = np.append(im, g, axis=2)
            x = np.append(x, [im], axis=0)
            # print("#############################")
            # print("x: " + str(x.shape))
            # print("g: " + str(g.shape))
            # print('###############################')
            # x = np.append(x, g, axis=0)
            y[i] = train_labels.at[sample, labels.get(0)]

        x = x[1:, 100:200, 100:200, :]


        y = y.reshape(self.batch_size, 1)
        y = keras.utils.to_categorical(y, num_classes=2)

        return x, y

# ------------------------- Define Model -----------------------------
ax0range=10; ax1range=100; ax2range=100; ax3range=4;
categories = 2;


lrs = [math.pow(10, i) for i in range(-2, 1, 1)]
momentums = [.1, .9]
lr = [.01]
momentums = [.1]
now = datetime.datetime.now()

print(lrs)
print(momentums)

csvfile = open(str(now.day)+str(now.hour) + str(now.min) + 'eggs.csv', 'w', newline='')
head = ['type', 'learning rate', 'momentum', 'epoch 1', 'epoch 2', ' ... ']
spamwriter = csv.writer(csvfile, delimiter=';',
                        quotechar='"', quoting=csv.QUOTE_MINIMAL)
spamwriter.writerow(head)
for lr in lrs:
    for m in momentums:
        print("-------" + str(lr) + " : " + str(m) + "--------")
        model = Sequential()
        # input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
        # this applies 32 convolution filters of size 3x3 each.
        model.add(Conv2D(100, (5, 5), activation='relu', input_shape=(ax1range, ax2range, ax3range)))
        model.add(MaxPooling2D(pool_size=(5, 5)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(BatchNormalization(axis=1))

        model.add(Dense(categories, activation='softmax'))
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd)

        # ------------------------ Fit the Model -------------------------------
        print("Number of samples available: " + str(len(train_labels)))
        train_history = model.fit_generator(generator = CIFAR10Sequence(train_labels=train_labels[0:28], batch_size=2),
                            steps_per_epoch = 14,
                            epochs = 5,
                            validation_data = CIFAR10Sequence(train_labels=train_labels[28:31], batch_size=1),
                            validation_steps = 3)

        # -----------------------record the results---------------------------
        losses = train_history.history['loss']
        val_losses = train_history.history['val_loss']
        spamwriter.writerow(["train", lr, m] + losses)
        spamwriter.writerow(["valid", lr, m] + val_losses)
csvfile.close()

with open("models/model.json", "w") as json_file:
    json_model = model.to_json()
    json_file.write(json_model)
model.save('modelWeights/weights')

# with open("models/model.json", "r") as json_file:
#     json_model = json_file.read()
#     model = model_from_json(json_model)
# model.load_weights('modelWeights/weights')


# ----------------------------- Test Data -------------------------------------

# test_generator = CIFAR10Sequence(train_labels=train_labels[100:140], batch_size=40)
# x_test, y_test = test_generator.__getitem__(1);
# print("The generated test data has mean: " + str(x_test.mean()))
# print("The generated test data has std: " + str(np.std(x_test)))
# print("The generated test data has shape: " + str(x_test.shape))
# print("Test generator batches: " + str(test_generator.__len__()));
#
#
# y_pred = model.predict(x_test)
#
# print("Predictions")
# print(y_pred)
# print("Actuals")
# print(y_test)

