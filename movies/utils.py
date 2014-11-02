import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_data(filename):
    return np.array(pd.read_csv(filename))


def rmse(a, b):
    """ Returns RMSE between two lists. """
    if not a or not b:
        return 0
    return (sum([(i - j) ** 2 for i, j in zip(a, b)]) / float(len(a))) ** 0.5


def get_learning_curves(model, train, test):
    random.shuffle(train)
    learning_curves = []
    start = 0
    # Learning curves on 0..train_set/10
    while 2 ** start < len(train)/10:
        curr_range = 2 ** start
        curr_train = train[:curr_range]
        print "Training on %s data..." % curr_range
        model.train(curr_train)
        lc_train = model.test(curr_train)
        lc_test = model.test(test)
        percentage = len(curr_train)/float(len(train)) * 100
        learning_curves.append((lc_train, lc_test, percentage))
        start += 1

    # Learning curves on train_set/10..train_set
    for i in range(1, 11):
        curr_range = len(train) * i / 10
        curr_train = train[:curr_range]
        print "Training on %s data..." % curr_range
        model.train(curr_train)
        lc_train = model.test(curr_train)
        lc_test = model.test(test)
        percentage = len(curr_train)/float(len(train)) * 100
        learning_curves.append((lc_train, lc_test, percentage))

    return learning_curves


def plot_learning_curves(learning_curves, title=None, axis=[0, 100, 0.5, 2.1]):
    print "Plotting learning curves..."
    plt.plot([y for _, _, y in learning_curves],
             [x for x, _, _ in learning_curves], label="Train set")
    plt.plot([y for _, _, y in learning_curves],
             [x for _, x, _ in learning_curves], label="Test set")
    plt.ylabel("RMSE")
    plt.xlabel("% of training data")
    if title:
        plt.title(title)
    if axis:
        plt.axis(axis)
    plt.legend()
    plt.show()


def plot_movies(X, movies, picked_movies, title=None):
    print "Plotting movies..."
    processed_movies = []
    for i, t, _ in movies:
        if i in picked_movies:
            processed_movies.append((i - 1, t))

    x = [X[i, 0] for i, _ in processed_movies]
    y = [X[i, 1] for i, _ in processed_movies]
    plt.plot(x, y, "ro")
    for i, t in processed_movies:
        x_shift = X[i, 0] + (max(x) - min(x))/20
        y_shift = X[i, 1] + (max(y) - min(y))/20
        plt.annotate(t, xy=(X[i, 0], X[i, 1]),
                     xytext=(x_shift, y_shift),
                     arrowprops=dict(arrowstyle="-"))

    if title:
        plt.title(title)
    plt.show()


def project_data(X, k):
    m = len(X)
    cov_matrix = np.dot(np.transpose(X), X)/m
    U, _, _ = np.linalg.svd(cov_matrix)
    U_r = U[:, :k]

    return np.dot(X, U_r)
