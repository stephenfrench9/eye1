import app
import numpy as np

if __name__ == '__main__':
    train_labels = app.data()
    modelOfInterest = "showers/"
    model = app.loadModel(modelOfInterest)

    # load test data
    batch_size = 30
    start = 28000
    test_generator = app.ImageSequence(train_labels=train_labels, batch_size=batch_size, start=start)
    # x_test, y_act = test_generator.__getitem__(0);
    # y_pred = model.predict(x_test)
    #
    # print(np.round(y_pred, 0))
    #
    # print(type(np.round(y_pred[-2][1], 0)))
    # print(np.round(y_pred[-1][1], 1))
    #
    # a = np.zeros((2, 2))
    #
    # for i in range(y_pred.shape[0]):
    #     prediction = int(np.round(y_pred[i][1], 0))
    #     actual = int(np.round(y_act[i][1], 0))
    #     a[prediction][actual] += 1
    #
    #
    # notes = "train.csv " + str(batch_size) + str(start)
    # app.writePerformance("showers/", a, notes)

    a = np.zeros((2, 2))
    num_batches = test_generator.__len__()
    for batch in range(num_batches-1):
        x_test, y_act = test_generator.__getitem__(batch)
        y_pred = model.predict(x_test)

        print(str(batch) + " out of " + str(num_batches))
        for i in range(y_pred.shape[0]):
            prediction = int(np.round(y_pred[i][1], 0))
            actual = int(np.round(y_act[i][1], 0))
            a[prediction][actual] += 1

    class1 = a[0][1] + a[1][1]
    recall = a[1][1] / class1

    exclaim = a[1][0] + a[1][1]
    precision = a[1][1] / exclaim

    print(a)
    print("precision: " + str(precision))
    print("recall: " + str(recall))
    notes = ["train.csv ", str(batch_size), str(start)]
    precision = ["precision: ", str(precision)]
    recall = ["recall: ", str(recall)]
    app.writePerformance(modelOfInterest, a, precision, recall, notes)
