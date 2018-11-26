import app

if __name__ == '__main__':
    train_labels = app.data()

    seq = app.ImageSequence(train_labels=train_labels[0:100], batch_size=10, start=0)

    x, y = seq.__getitem__(0)

    print(x.shape)
    print(y.shape)
    print(y)
