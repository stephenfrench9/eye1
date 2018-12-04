import app

if __name__ == "__main__":
    print("search is running")

    lrs = [.1 ,1]
    ms = [0, .5]
    neurons = [2, 10]
    filters = [4, 10]

    train_labels = app.data()

    app.search_parameters(lrs, ms, neurons, filters, train_labels=train_labels)