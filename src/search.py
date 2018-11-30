import app

if __name__ == "__main__":
    print("search is running")

    lrs = [.1, 1]
    ms = [0, .9]
    neurons = [100, 10]

    train_labels = app.data()

    app.search_parameters(lrs, ms, neurons, train_labels=train_labels)