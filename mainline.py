#!/usr/bin/env python

from movies import models
from movies.utils import load_data, get_learning_curves, plot_learning_curves

if __name__ == "__main__":
    train = load_data("ratings-train.csv")
    test = load_data("ratings-test.csv")
    print "Training the baseline predictor on %s data points..." % len(train)
    m = models.BaseModel(maxiter=100, n_features=9, l=8)
    #m.train(train)
    #print "The test set contains %s data points." % len(test)
    #print "RMSE on the test set is %.4f" % m.test(test)

    # Learning curves
    learning_curves = get_learning_curves(m, train, test)
    print "Learning curves: \n%s" % str(learning_curves)
    plot_learning_curves(learning_curves)
