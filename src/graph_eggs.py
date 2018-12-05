import csv
import matplotlib.pyplot as plt
if __name__ == '__main__':
    root = "/ralston/"
    model_of_interest = "4-18-36/"

    print("graph eggs is running")

    with open(root + "models/" + model_of_interest + "eggs.csv", 'r', newline='') as f:
        rows = csv.reader(f, delimiter=';')
        for row in rows:
            if row[0] == "train":
                plt.plot([float(i)/float(row[1]) for i in row[1:]], 'r')
            elif row[0] == "valid":
                plt.plot([float(i)/float(row[1]) for i in row[1:]], 'b')
            elif row[0] == "pred_1":
                plt.plot([float(i) for i in row[1:]], 'y')
            elif row[0] == "act_1":
                plt.plot([float(i) for i in row[1:]], 'g')
            plt.title("train=red, validation=blue,\npred_yes=yellow, act_yes=g")
            plt.xlabel("epoch")
            plt.savefig(root + "models/" + model_of_interest + "eggs.png")
