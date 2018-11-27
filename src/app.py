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

root = "/ralston/"


def data():
    train_labels = pd.read_csv(root + "train.csv")
    labels = {
        0: "Nucleoplasm",
        1: "Nuclear membrane",
        2: "Nucleoli",
        3: "Nucleoli fibrillar center",
        4: "Nuclear speckles",
        5: "Nuclear bodies",
        6: "Endoplasmic reticulum",
        7: "Golgi apparatus",
        8: "Peroxisomes",
        9: "Endosomes",
        10: "Lysosomes",
        11: "Intermediate filaments",
        12: "Actin filaments",
        13: "Focal adhesion sites",
        14: "Microtubules",
        15: "Microtubule ends",
        16: "Cytokinetic bridge",
        17: "Mitotic spindle",
        18: "Microtubule organizing center",
        19: "Centrosome",
        20: "Lipid droplets",
        21: "Plasma membrane",
        22: "Cell junctions",
        23: "Mitochondria",
        24: "Aggresome",
        25: "Cytosol",
        26: "Cytoplasmic bodies",
        27: "Rods & rings"
    }

    reverse_train_labels = dict((v, k) for k, v in labels.items())

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
            0: "Nucleoplasm",
            1: "Nuclear membrane",
            2: "Nucleoli",
            3: "Nucleoli fibrillar center",
            4: "Nuclear speckles",
            5: "Nuclear bodies",
            6: "Endoplasmic reticulum",
            7: "Golgi apparatus",
            8: "Peroxisomes",
            9: "Endosomes",
            10: "Lysosomes",
            11: "Intermediate filaments",
            12: "Actin filaments",
            13: "Focal adhesion sites",
            14: "Microtubules",
            15: "Microtubule ends",
            16: "Cytokinetic bridge",
            17: "Mitotic spindle",
            18: "Microtubule organizing center",
            19: "Centrosome",
            20: "Lipid droplets",
            21: "Plasma membrane",
            22: "Cell junctions",
            23: "Mitochondria",
            24: "Aggresome",
            25: "Cytosol",
            26: "Cytoplasmic bodies",
            27: "Rods & rings"
        }

    def __len__(self):
        return int(np.ceil((len(self.train_labels)) / float(self.batch_size)))-1

    def __getitem__(self, idx):
        trials = len(self.train_labels)
        y = np.ones((self.batch_size, 1))
        x = np.ones((1, 512, 512, 4))

        # y = to_cate

        # x = np.ones((1, 512, 512))
        for i in range(self.batch_size):
            sample = self.start + i + idx * self.batch_size
            b = imread(self.base + self.train_labels.at[sample, 'Id'] + self.red).reshape((512, 512, 1))
            r = imread(self.base + self.train_labels.at[sample, 'Id'] + self.blue).reshape((512, 512, 1))
            ye = imread(self.base + self.train_labels.at[sample, 'Id'] + self.yellow).reshape((512, 512, 1))
            g = imread(self.base + self.train_labels.at[sample, 'Id'] + self.green).reshape((512, 512, 1))
            im = np.append(b, r, axis=2)
            im = np.append(im, ye, axis=2)
            im = np.append(im, g, axis=2)
            x = np.append(x, [im], axis=0)
            # y[i] = self.train_labels.at[sample, self.labels.get(0)]
            g = self.train_labels.ix[sample]
            y[i, :] = np.array(g[2:3])

        x = x[1:, 100:200, 100:200, :]
        # plt.imsave("/ralston/pictures/blue.png", x[0][:, :, 0])
        # plt.imsave("/ralston/pictures/red.png", x[0][:, :, 1])
        # plt.imsave("/ralston/pictures/yellow.png", x[0][:, :, 2])
        # plt.imsave("/ralston/pictures/green.png", x[0][:, :, 3])

        # y = y.reshape(self.batch_size, 1)
        y = keras.utils.to_categorical(y, num_classes=2)

        max = np.max(x)
        max = abs(max)
        x /= max
        mean = np.mean(x)
        x -= mean

        o = x.shape
        # print(o)
        # x = x.reshape(o[0], o[1] * o[2] * o[3])

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


def model3(lrp, mp):
    ax0range = 10;
    ax1range = 40000;
    categories = 28;

    model = Sequential()
    model.add(Dense(1000, activation='relu', input_dim=40000))
    model.add(Dense(categories, activation='softmax'))
    sgd = SGD(lr=lrp, decay=1e-6, momentum=mp, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model


def model4(lrp, mp):
    """destined to fail - you cant predict rare categories with common ones.
    just produces straight zeros. only predicts zeros for everything"""
    ax0range = 10;
    ax1range = 100;
    ax2range = 100;
    ax3range = 4;
    categories = 28;

    model = Sequential()
    model.add(Conv2D(2, (5, 5), activation='relu', input_shape=(ax1range, ax2range, ax3range)))
    model.add(MaxPooling2D(pool_size=(7, 7)))
    # model.add(Conv2D(4, (5, 5), activation='relu'))
    # model.add(Dropout(0.25))
    model.add(Flatten())
    # model.add(BatchNormalization(axis=1))
    model.add(Dense(categories, activation='softmax'))
    sgd = SGD(lr=lrp, decay=1e-6, momentum=mp, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model


def model5(lrp, mp):
    """only predict one category at a time"""
    ax0range = 10;
    ax1range = 100;
    ax2range = 100;
    ax3range = 4;
    categories = 2;

    model = Sequential()
    model.add(Conv2D(2, (5, 5), activation='relu', input_shape=(ax1range, ax2range, ax3range)))
    model.add(MaxPooling2D(pool_size=(7, 7)))
    # model.add(Conv2D(4, (5, 5), activation='relu'))
    # model.add(Dropout(0.25))
    model.add(Flatten())
    # model.add(BatchNormalization(axis=1))
    model.add(Dense(categories, activation='softmax'))
    sgd = SGD(lr=lrp, decay=1e-6, momentum=mp, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model


def loadModel(model):
    """
    :return: a model
    """
    destination = root + "models/" + model
    with open(destination + "model.json", "r") as json_file:
        json_model = json_file.read()
        model = model_from_json(json_model)
    model.load_weights(destination + 'weights')

    return model


def writePerformanceSingle(model, cm, precision, recall, notes):
    """
    try out a model on some test data
    :param model: the NAME of the model
    :param cm: the confusion matrix
    :param notes: typically the test data
    """
    destination = root + "models/" + model
    with open(destination + "performance.csv", "w") as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=';',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header = ["           ", "actual 0", "actual 1"]
        row1 = ["predicted 0", str(int(cm[0][0])), str(int(cm[0][1]))]
        row2 = ["predicted 1", str(int(cm[1][0])), str(int(cm[1][1]))]
        csvwriter.writerow(header)
        csvwriter.writerow(row1)
        csvwriter.writerow(row2)
        csvwriter.writerow(precision)
        csvwriter.writerow(recall)
        csvwriter.writerow(notes)


def writePerformanceMulti(model, precisions, recalls, notes):
    """
    Write to a csv the precisions and recalls for every class
    :param model: the NAME of the model
    :param precisions: all the precisions for all the difference classes
    :param notes: typically the test data
    """
    destination = root + "models/" + model
    with open(destination + "performance.csv", "a") as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=';',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header = ["Class Number", "Precision", "Recall"]
        csvwriter.writerow(header)

        for i in range(28):
            i = i +1
            csvwriter.writerow([str(i), precisions[i], recalls[i]])

        csvwriter.writerow(notes)


def search_parameters(lrs, momentums, train_labels):
    now = datetime.datetime.now()
    modelID = str(now.day) + str(now.hour) + str(now.minute) + str(now.second)
    destination = root + "models/" + modelID + "/"
    if not os.path.isdir(destination):
        os.mkdir(destination)
    csvfile = open(destination + 'eggs.csv', 'w', newline='')
    head = ['type', 'learning rate', 'momentum', 'epoch 1', 'epoch 2', ' ... ']
    spamwriter = csv.writer(csvfile, delimiter=';',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(head)
    for lr in lrs:
        for m in momentums:
            model = model5(lr, m)
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


def writeCsv(csvfile, train_history, lr, p, train_l, train_h, train_batch_size, valid_l, valid_h, valid_batch_size,
             modelName):

    head = ['type', 'epoch 1', 'epoch 2', ' ... ']
    spamwriter = csv.writer(csvfile, delimiter=';',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)

    spamwriter.writerow(head)
    losses = train_history.history['loss']
    val_losses = train_history.history['val_loss']
    spamwriter.writerow(["train"] + losses)
    spamwriter.writerow(["valid"] + val_losses)
    spamwriter.writerow([" ... "])
    spamwriter.writerow(["train",
                         "train_labels: " + str(train_l) + ":" + str(train_h),
                         "batch_size: " + str(train_batch_size),
                         "learning rate: " + str(lr),
                         "momentum: " + str(p),
                         "model name: " + modelName])

    spamwriter.writerow(["test",
                         "test_labels: " + str(valid_l) + ":" + str(valid_h),
                         "batch_size: " + str(valid_batch_size)])


if __name__ == "__main__":
    # load the data
    train_labels = data()

    # train a model
    lr = .1
    p = .1
    model = model5(lr, p)

    train_l = 0; train_h = 28000;
    train_batch_size = 2
    train_batches = train_h/train_batch_size

    valid_l = 28000; valid_h = 31000;
    valid_batch_size = 2 # valid_batch_size =10 and valid_batches = 1 does not work ... cra
    valid_batches = (valid_h-valid_l)/valid_batch_size

    train_history = model.fit_generator(generator=ImageSequence(train_labels[train_l:train_h],
                                                                batch_size=train_batch_size,
                                                                start=train_l),
                                        steps_per_epoch=train_batches,
                                        epochs=15,
                                        validation_data=ImageSequence(train_labels[valid_l:valid_h],
                                                                      batch_size=valid_batch_size,
                                                                      start=valid_l),
                                        validation_steps=valid_batches)

    # save stuff
    now = datetime.datetime.now()
    modelID = str(now.day) + str(now.hour) + str(now.minute) + str(now.second)
    destination = root + "models/" + modelID + "/"
    if not os.path.isdir(destination):
        os.mkdir(destination)

    with open(destination + 'eggs.csv', 'w', newline='') as csvfile:
        writeCsv(csvfile, train_history, lr, p, train_l, train_h, train_batch_size, valid_l, valid_h, valid_batch_size,
                 "model2")

    with open(destination + "model.json", "w") as json_file:
        json_model = model.to_json()
        json_file.write(json_model)

    model.save(destination + "weights")



