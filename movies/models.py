import numpy as np

from scipy import optimize

from utils import rmse, project_data


class AbstractModel(object):
    def __init__(self, n_movies=3952, n_users=6040, n_features=1):
        self.n_movies = n_movies
        self.n_users = n_users
        self.n_features = n_features

    def init_data(self, data):
        X = np.random.random([self.n_movies, self.n_features])
        Theta = np.random.random([self.n_users, self.n_features])
        Y = np.zeros([self.n_movies, self.n_users])
        R = np.zeros([self.n_movies, self.n_users])

        for movie, user, rating in data:
            if movie-1 < self.n_movies and user-1 < self.n_users:
                Y[movie-1, user-1] = rating
                R[movie-1, user-1] = 1

        return X, Theta, Y, R

    def to_params(self, X, Theta):
        return np.concatenate([X.flatten(), Theta.flatten()])

    def from_params(self, params):
        X = params[0:self.n_movies * self.n_features].\
            reshape(self.n_movies, self.n_features)
        Theta = params[self.n_movies * self.n_features:].\
            reshape(self.n_users, self.n_features)
        return X, Theta

    def train(self, data):
        raise NotImplementedError()

    def test(self, data):
        raise NotImplementedError()


class BaseModel(AbstractModel):
    def __init__(self, n_movies=3952, n_users=6040, n_features=1,
                 maxiter=50, l=0, d=None):
        self.n_movies = n_movies
        self.n_users = n_users
        self.n_features = n_features
        self.maxiter = maxiter
        self.l = l
        self.d = d

    def cost_func(self, params, Y, R):
        # Helper params
        X, Theta = self.from_params(params)
        Theta_t = np.transpose(Theta)

        # Compute cost
        rmse = ((np.dot(X, Theta_t) - Y) ** 2) * R
        cost = np.sum(rmse) / 2.0

        # Regularization
        cost += self.l/2.0 * np.sum(X ** 2) + self.l/2.0 * np.sum(Theta ** 2)

        return cost

    def grad_func(self, params, Y, R):
        # Helper params
        X, Theta = self.from_params(params)
        Theta_t = np.transpose(Theta)

        # Compute gradient
        sub = (np.dot(X, Theta_t) - Y) * R
        X_grad = np.dot(sub, Theta)
        Theta_grad = np.dot(np.transpose(sub), X)

        # Regularization
        X_grad += self.l * X
        Theta_grad += self.l * Theta

        return self.to_params(X_grad, Theta_grad)

    def reduce_dimensions(self, d=None):
        if not d and not self.d:
            return
        elif not d:
            d = self.d
        X_Theta = np.concatenate([self.X, self.Theta])
        X_Theta_d = project_data(X_Theta, d)
        self.X = X_Theta_d[:self.n_movies]
        self.Theta = X_Theta_d[self.n_movies:]

        self.predictions = np.dot(self.X, np.transpose(self.Theta))

    def train(self, data):
        X, Theta, Y, R = self.init_data(data)
        params = self.to_params(X, Theta)
        result = optimize.fmin_cg(self.cost_func, params,
                                  fprime=self.grad_func,
                                  args=(Y, R), maxiter=self.maxiter)
        self.X, self.Theta = self.from_params(result)
        self.reduce_dimensions()

        self.predictions = np.dot(self.X, np.transpose(self.Theta))

    def test(self, data):
        predicted = []
        real = []
        for movie, user, rating in data:
            if movie-1 < self.n_movies and user-1 < self.n_users:
                predicted.append(self.predictions[movie-1, user-1])
                real.append(rating)
        return rmse(real, predicted)
