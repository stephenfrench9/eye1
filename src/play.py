import app



if __name__ == '__main__':

    train_labels = app.data()
    y_pred = app.load_and_predict("1922299/", train_labels[0:100], 0)

    print(y_pred)

