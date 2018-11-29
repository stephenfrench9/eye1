import csv
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    root = "/ralston/"
    searchOfInterest = "29-22-19/"

    csv_file = open(root + "searches/" + searchOfInterest + "eggs.csv", 'r', newline='')

    reader = csv.reader(csv_file, delimiter=";")

    i = 0
    modelName = "model"
    for row in reader:
        if row[0] == "type":
            a = 1
        elif row[0] == "test_data":
            a = 1
        elif row[0] == "train_data":
            a = 1
            modelName = row[-1]
        elif row[0] == "accuracy":
            a = 1
        elif row[0] == "train" or row[0] == "valid":
            result = [float(i)/float(row[3]) for i in row[3:]]
            if row[0] == "train":
                plt.plot(result, "r-")
            elif row[0] == "valid":
                plt.plot(result, "b")
            if row[0] == "valid":
                plt.title(modelName + "\n" + "lr: " + row[1] + " momentum: " + row[2])
                plt.xlabel("epoch -- train = red, validation = blue")
                plt.ylabel("loss, %")
                plt.ylim([.2, 1.2])
                plt.savefig(root + "pictures/" + str(i) + ".png")
                plt.clf()
        i += 1

