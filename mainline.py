#!/usr/bin/env python

from movies import models
from movies.utils import load_data

if __name__ == "__main__":
    train = load_data("ratings-train.csv")
    test = load_data("ratings-test.csv")
    print "Training the baseline predictor on %s data points..." % len(train)
    m = models.BaseModel(maxiter=120)
    m.train(train)
    print "The test set contains %s data points." % len(test)
    print "RMSE on the test set is %.4f" % m.test(test)
