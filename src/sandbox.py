import app
import matplotlib.pyplot as plt

if __name__ == '__main__':
    train_labels = app.data()

    seq = app.ImageSequence(train_labels=train_labels[0:100], batch_size=10, start=0)

    x, y = seq.__getitem__(0)

    x = x.reshape(10, 100, 100, 4)

    im = plt.imsave("/ralston/pictures/hooligan.png", x[0][:, :, 0])

    print(x.shape)
    print(y.shape)
    print(y)
