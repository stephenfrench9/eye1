import os
from keras.models import Sequential
from os import listdir
import keras
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
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
        for i in range(self.batch_size):
            sample = i + idx * self.batch_size
            b = imread(base + train_labels.at[sample, 'Id'] + red).reshape((512, 512, 1))
            r = imread(base + train_labels.at[sample, 'Id'] + red).reshape((512, 512, 1))
            ye = imread(base + train_labels.at[sample, 'Id'] + yellow).reshape((512, 512, 1))
            g = imread(base + train_labels.at[sample, 'Id'] + green).reshape((512, 512, 1))
            im = np.append(b, r, axis=2)
            im = np.append(im, ye, axis=2)
            im = np.append(im, g, axis=2)
            x = np.append(x, [im], axis=0)
            y[i] = train_labels.at[sample, labels.get(0)]

        x = x[1:, :, :, :]
        y = y.reshape(self.batch_size, 1)
        y = keras.utils.to_categorical(y, num_classes=2)

        return x, y

# ------------------------- Define Model -----------------------------
ax0range=10; ax1range=512; ax2range=512; ax3range=4;
categories = 2;

model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(100, (5, 5), activation='relu', input_shape=(ax1range, ax2range, ax3range)))
# model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(5, 5)))
model.add(Dropout(0.25))

# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(categories, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd)


# ------------------------ Fit the Model -------------------------------

model.fit_generator(generator = CIFAR10Sequence(train_labels=train_labels[0:200], batch_size=10),
                    steps_per_epoch = 200,
                    epochs = 10,
                    validation_data = CIFAR10Sequence(train_labels=train_labels[200:220], batch_size=5),
                    validation_steps = 20)


