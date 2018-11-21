import app
import numpy as np

if __name__ == '__main__':
    train_labels = app.data()
    modelOfInterest = "ants/"
    model = app.loadModel(modelOfInterest)

    # load test data
    batch_size = 30
    valid_l = 30500; valid_h = 31000
    test_generator = app.ImageSequence(train_labels=train_labels[valid_l:valid_h],
                                       batch_size=batch_size,
                                       start=valid_l)
    # initialize confusion matrix
    a = np.zeros((2, 2))

    # build confusion matrix
    num_batches = test_generator.__len__()
    for batch in range(num_batches):
        x_test, y_act = test_generator.__getitem__(batch)
        y_pred = model.predict(x_test)

        print(str(batch) + " out of " + str(num_batches))
        for i in range(y_pred.shape[0]):
            prediction = int(np.round(y_pred[i][1], 0))
            actual = int(np.round(y_act[i][1], 0))
            a[prediction][actual] += 1

    # calculate performance metrics
    class1 = a[0][1] + a[1][1]
    recall = a[1][1] / class1
    exclaim = a[1][0] + a[1][1]
    precision = a[1][1] / exclaim

    # save the results
    notes = ["file: train.csv", "range: " + str(valid_l) + ":" + str(valid_h), "batch size: " + str(batch_size)]
    precision = ["precision: ", str(precision)]
    recall = ["recall: ", str(recall)]
    app.writePerformance(modelOfInterest, a, precision, recall, notes)
