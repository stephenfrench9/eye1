import csv
import matplotlib.pyplot as plt
if __name__ == '__main__':
    root = "./"
    model_of_interest = "7-11-46-35/"

    print("graph eggs is running")

    with open(root + "models/" + model_of_interest + "eggs.csv", 'r', newline='') as f:
        rows = csv.reader(f, delimiter=';')
        for row in rows:
            x = [i+1 for i in range(len(row[1:]))]

            if row[0] == "train":
                plt.plot(x, [float(i)/float(row[1]) for i in row[1:]], 'r')
            elif row[0] == "valid":
                plt.plot(x, [float(i)/float(row[1]) for i in row[1:]], 'b')
            elif row[0] == "pred_1":
                plt.plot(x, [float(i) for i in row[1:]], 'y')
            elif row[0] == "act_1":
                plt.plot(x, [float(i) for i in row[1:]], 'g')
            plt.ylabel("normalized loss")
            plt.xlabel("epoch : train=red, validation=blue, pred_yes=yellow, act_yes=g")
            # plt.xticks(x)
            plt.title("model 6 (" + model_of_interest + ")\nlr=.1, m=0, N=10,f=10")
            plt.savefig(root + "models/" + model_of_interest + "eggs.png")
