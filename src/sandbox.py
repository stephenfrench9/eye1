import app
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    train_labels = app.data()

    seq = app.ImageSequence(train_labels=train_labels[0:100], batch_size=50, start=0)

    x, y = seq.__getitem__(0)

    model = app.model5(.1, .1)

    y_pred = model.predict(x)

    print("actual: ")
    print(y)
    print("prediction: ")
    print(y_pred)





