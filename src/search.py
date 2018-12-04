import app

if __name__ == "__main__":
    print("search is running")

    lrs = [1]
    ms = [0]
    neurons = [5]
    filters = [10]

    train_labels = app.data()

    app.search_parameters(lrs, ms, neurons, filters, train_labels=train_labels)