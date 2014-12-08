#!/usr/bin/env python

from movies import models, utils

if __name__ == "__main__":
    train = utils.load_data("ratings-train.csv")
    test = utils.load_data("ratings-test.csv")

    # Configuration
    features = 9
    l = 0
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
        356,   # Forrest Gump
        924,   # 2001: A Space Odyssey
        1198,  # Raiders of the Lost Arch
        1206,  # Clockwork Orange, A
        1291,  # Indiana Jones and the Last Crusade
        1721,  # Titanic
        2115,  # Indiana Jones and the Last Crusade
        2116,  # Lord of the Rings, The
        2571,  # Matrix
        3535,  # American Psycho
        1676,  # Starship Troopers (1997)
    ]
    #utils.plot_movies(m.X, movies, picked_movies, title=title)

    genres = {'Action', 'Adventure', 'Animation', "Children's",
              'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
              'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
              'Sci-Fi', 'Thriller', 'War', 'Western'}

    kubrick = [2727, 2726, 1178, 2728, 2729, 750, 924, 1206, 2730, 1258, 1222, 2712]
    spielberg = [1693, 2028, 480, 527, 2115, 1291, 1198, 1097, 1387, 3471]
    fincher = [1320, 47, 1625, 2959]
    scorsese = [111, 1213, 1343, 16, 1228, 412, 2022, 2976, 1730]
    tarantino = [1089, 555, 288, 296, 18, 1729]


    # Learning curves
    #learning_curves = utils.get_learning_curves(m, train, test)
    #print "Learning curves: \n%s" % str(learning_curves)
    #utils.plot_learning_curves(learning_curves, title=title)
