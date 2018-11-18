import csv
import datetime
import keras
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.models import Sequential, model_from_json
from keras.optimizers import SGD
from keras.utils import Sequence
from scipy.misc import imread
from skimage.io import imread
warnings.filterwarnings("ignore", category=DeprecationWarning)


def data():
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

    for key in labels.keys():
        train_labels[labels[key]] = 0

    def fill_targets(row):
        row.Target = np.array(row.Target.split(" ")).astype(np.int)
        for num in row.Target:
            name = labels[int(num)]
            row.loc[name] = 1
        return row

    train_labels = train_labels.apply(fill_targets, axis=1)

    return train_labels


class ImageSequence(Sequence):

    def __init__(self, train_labels, batch_size, start):
        self.train_labels = train_labels
        self.batch_size = batch_size
        self.base = "train/"
        self.blue = "_blue.png"
        self.red = "_red.png"
        self.yellow = "_yellow.png"
        self.green = "_green.png"
        self.start = start
        self.labels = {
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

    def __len__(self):
        return int(np.ceil(len(self.train_labels) / float(self.batch_size)))

    def __getitem__(self, idx):
        trials = len(self.train_labels)
        y = np.ones(self.batch_size)
        x = np.ones((1, 512, 512, 4))

        # x = np.ones((1, 512, 512))
        for i in range(self.batch_size):
            sample = self.start + i + idx * self.batch_size
            b = imread(self.base + self.train_labels.at[sample, 'Id'] + self.red).reshape((512, 512, 1))
            b = b / np.std(b)
            b = b - b.mean()
            r = imread(self.base + self.train_labels.at[sample, 'Id'] + self.blue).reshape((512, 512, 1))
            r = r / np.std(r)
            r = r - r.mean()
            ye = imread(self.base + self.train_labels.at[sample, 'Id'] + self.yellow).reshape((512, 512, 1))
            ye = ye / np.std(ye)
            ye = ye - ye.mean()
            g = imread(self.base + self.train_labels.at[sample, 'Id'] + self.green).reshape((512, 512, 1))
            g = g / np.std(g)
            g = g - g.mean()
            im = np.append(b, r, axis=2)
            im = np.append(im, ye, axis=2)
            im = np.append(im, g, axis=2)
            x = np.append(x, [im], axis=0)
            y[i] = self.train_labels.at[sample, self.labels.get(0)]

        x = x[1:, 100:200, 100:200, :]


        y = y.reshape(self.batch_size, 1)
        y = keras.utils.to_categorical(y, num_classes=2)

        max = np.max(x)
        max = abs(max)
        x /= max
        mean = np.mean(x)
        x -= mean

        o = x.shape
        x = x.reshape(o[0], o[1] * o[2] * o[3])

        return x, y


def model0(lrp, mp):
    ax0range = 10;
    ax1range = 100;
    ax2range = 100;
    ax3range = 4;
    categories = 2;

    model = Sequential()
    model.add(Conv2D(100, (5, 5), activation='relu', input_shape=(ax1range, ax2range, ax3range)))
    model.add(MaxPooling2D(pool_size=(5, 5)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(BatchNormalization(axis=1))
    model.add(Dense(categories, activation='softmax'))
    sgd = SGD(lr=lrp, decay=1e-6, momentum=mp, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model


def model1(lrp, mp):
    ax0range = 10;
    ax1range = 40000;
    categories = 2;

    model = Sequential()
    model.add(Dense(2000, activation='relu', input_dim=40000))
    model.add(Dense(categories, activation='softmax'))
    sgd = SGD(lr=lrp, decay=1e-6, momentum=mp, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model


def model2(lrp, mp):
    ax0range = 10;
    ax1range = 40000;
    categories = 2;

    model = Sequential()
    model.add(Dense(1000, activation='relu', input_dim=40000))
    model.add(Dense(categories, activation='softmax'))
    sgd = SGD(lr=lrp, decay=1e-6, momentum=mp, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model


def load_and_predict(train_labels, start):
    """
    :param train_labels: loads the model that is in models/ and the
    weights from modelWeights/, and the training datas that you pass into it.
    :param start: an index in the original train labels. Tells you the first row
    present in this data set
    :return:
    """
    with open("models/model.json", "r") as json_file:
        json_model = json_file.read()
        model = model_from_json(json_model)
    model.load_weights('modelWeights/weights')

    # load test data
    test_generator = ImageSequence(train_labels=train_labels, batch_size=40, start=start)
    x_test, y_test = test_generator.__getitem__(0);
    print("The generated test data has mean: " + str(x_test.mean()))
    print("The generated test data has std: " + str(np.std(x_test)))
    print("The generated test data has shape: " + str(x_test.shape))
    print("Test generator batches: " + str(test_generator.__len__()));

    y_pred = model.predict(x_test)

    print(y_pred)


def search_parameters(lrs, momentums):
    now = datetime.datetime.now()
    csvfile = open(str(now) + 'eggs.csv', 'w', newline='')
    head = ['type', 'learning rate', 'momentum', 'epoch 1', 'epoch 2', ' ... ']
    spamwriter = csv.writer(csvfile, delimiter=';',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(head)
    for lr in lrs:
        for m in momentums:
            model = model0(lr, m)
            # ------------------------ Fit the Model -------------------------------
            print("Number of samples available: " + str(len(train_labels)))
            train_history = model.fit_generator(
                generator=ImageSequence(train_labels=train_labels[0:28], batch_size=2, start=0),
                steps_per_epoch=14,
                epochs=5,
                validation_data=ImageSequence(train_labels=train_labels[28:31], batch_size=1, start=28),
                validation_steps=3)

            # -----------------------record the results---------------------------
            losses = train_history.history['loss']
            val_losses = train_history.history['val_loss']
            spamwriter.writerow(["train", lr, m] + losses)
            spamwriter.writerow(["valid", lr, m] + val_losses)
    csvfile.close()
    # with open("models/model.json", "w") as json_file:
    #     json_model = model.to_json()
    #     json_file.write(json_model)
    # model.save('modelWeights/weights')


if __name__=="__main__":
    print("this is the main activity")
    os.chdir("/ralston")
    print(os.listdir("."))

    # load the data
    train_labels = data()

    # parameter search
    # lrs = [math.pow(10, i) for i in range(0, 1, 1)]
    # momentums = [.1]
    # search_parameters(lrs, momentums)

    model = model2(.1, .1)

    model.fit_generator(generator=ImageSequence(train_labels[0:28000], batch_size=10, start=0),
                        steps_per_epoch=2800,
                        epochs=15,
                        validation_data=ImageSequence(train_labels[28000:31000], batch_size=10, start=28000),
                        validation_steps=300)

    test_generator = ImageSequence(train_labels=train_labels[0:50], batch_size=40, start=0)
    x_test, y_test = test_generator.__getitem__(0);
    y_pred = np.round(model.predict(x_test), 2)
    print(y_pred)
    with open("models/model2.json", "w") as json_file:
        json_model = model.to_json()
        json_file.write(json_model)
    model.save('modelWeights/weights2')



    # load model
    # load_and_predict(train_labels[40:120], 40)
