import app
import matplotlib.pyplot as plt

if __name__ == '__main__':
    train_labels = app.data()

    seq = app.ImageSequence(train_labels=train_labels[0:100], batch_size=10, start=0)

    x, y = seq.__getitem__(0)

    print(x.shape)
    x = x.reshape(10, 100, 100, 4)

    model = app.model4(.1, .1)

    y_pred = model.predict(x)

    print("input: " + str(x.shape))
    print("output: " + str(y_pred.shape))


