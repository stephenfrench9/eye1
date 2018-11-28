import csv
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    print("changes")
    root = "/ralston/"
    searchOfInterest = "28-22-2/"

    csv_file = open(root + "searches/" + searchOfInterest + "eggs.csv", 'r', newline='')

    reader = csv.reader(csv_file, delimiter=";")

    header = True
    for row in reader:
        if header:
            a = 1
            header = False
        elif row[0] == '..':
            a = 1
            break
        else:
            result = row[:3]
            print(result)

            

            print(rounded)
            plt.plot(row[3:])
            plt.title("hank")
            plt.savefig(root + "pictures/" + "hoosan.png")
            plt.clf()


