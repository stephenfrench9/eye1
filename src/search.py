import app

if __name__ == "__main__":
    print("search is running")

    lrs = [.1, 1, 10]
    ms = [0, .1, .9]

    train_labels = app.data()


    app.search_parameters(lrs, ms, train_labels=train_labels)