import random
import matplotlib.pyplot as plt


def load_data(filename):
    """
    Returns a list of data points.
    Each data point is a list representing:
        [movie, user, rating]
    """
    with open(filename) as f:
        lines = f.readlines()
    lines.pop(0)  # pop the header
    return [map(int, line.split(',')) for line in lines]


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


def plot_learning_curves(learning_curves):
    print "Plotting learning curves..."
    plt.plot([y for _, _, y in learning_curves],
             [x for x, _, _ in learning_curves], label="Train set")
    plt.plot([y for _, _, y in learning_curves],
             [x for _, x, _ in learning_curves], label="Test set")
    plt.ylabel("RMSE")
    plt.xlabel("% of training data")
    plt.legend()
    plt.show()
