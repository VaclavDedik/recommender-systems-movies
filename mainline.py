#!/usr/bin/env python

from movies import models, utils

if __name__ == "__main__":
    train = utils.load_data("ratings-train.csv")
    test = utils.load_data("ratings-test.csv")

    # Configuration
    features = 9
    l = 8
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
    picked_movies = [
        1,     # Toy Story
        6,     # Heat
        #260,   # Star Wars Episode IV
        356,   # Forrest Gump
        364,   # Lion King, The
        858,   # Godfather
        924,   # 2001: A Space Odyssey
        1198,  # Raiders of the Lost Arch
        1206,  # Clockwork Orange, A
        #1219,  # Psycho
        1221,  # Godfather II
        1291,  # Indiana Jones and the Last Crusade
        1721,  # Titanic
        2115,  # Indiana Jones and the Last Crusade
        2116,  # Lord of the Rings, The
        2571,  # Matrix
        2858,  # American Beauty
        3527,  # Predator
        3535,  # American Psycho
    ]
    utils.plot_movies(m.X, movies, picked_movies, title=title)

    # Learning curves
    #learning_curves = utils.get_learning_curves(m, train, test)
    #print "Learning curves: \n%s" % str(learning_curves)
    #utils.plot_learning_curves(learning_curves, title=title)
