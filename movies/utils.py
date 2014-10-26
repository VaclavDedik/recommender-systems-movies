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
    return (sum([(i - j) ** 2 for i, j in zip(a, b)]) / float(len(a))) ** 0.5
