#!/usr/bin/env python

from movies import models, utils

if __name__ == "__main__":
    train = utils.load_data("ratings-train.csv")
    test = utils.load_data("ratings-test.csv")

    # Configuration
    features = 3
    l = 2
    maxiter = 100
    title = "features=%s, lambda=%s, maxiter=%s" % (features, l, maxiter)
    print "Training the baseline predictor on %s data points..." % len(train)
    m = models.BaseModel(maxiter=maxiter, n_features=features, l=l)
    m.train(train)
    print "The test set contains %s data points." % len(test)
    print "RMSE on the test set before projection is %.4f" % m.test(test)
    print "RMSE on the train set before projection is %.4f" % m.test(train)
    m.reduce_dimensions(2)
    print "RMSE on the test set after projection is %.4f" % m.test(test)
    print "RMSE on the train set after projection is %.4f" % m.test(train)

    # Plot movies visualization
    movies = utils.load_data("movies.csv")
    picked_movies = [356, 1198, 1291, 1721, 2115, 2116, 3535]
    utils.plot_movies(m.X, movies, picked_movies, title=title)

    # Learning curves
    #learning_curves = utils.get_learning_curves(m, train, test)
    #print "Learning curves: \n%s" % str(learning_curves)
    #utils.plot_learning_curves(learning_curves, title=title)
