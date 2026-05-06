import math

from src.util.BinaryTree import BCMOpt, gaussian_noise_variance
from src.util.Matrix import *


class PQFTL:
    def __init__(self, n, d, eps, delta, alpha, X_bound, y_bound):
        self.n = n
        self.d = d
        self.eps = eps
        self.delta = delta
        self.X_bound = X_bound
        self.y_bound = y_bound
        # Jain et al. originally uses same bound R for both X and y, but we allow different bounds here
        self.xtx = BCMOpt(n, eps / 2.0, d, d, gaussian_noise_variance(eps / 2.0, delta / 2.0,
                                                                      4.0 * math.log2(n) * math.pow(X_bound, 4)))
        self.xty = BCMOpt(n, eps / 2.0, d, 1, gaussian_noise_variance(eps / 2.0, delta / 2.0,
                                                                      4 * math.log2(n) * math.pow(X_bound,
                                                                                                  2) * math.pow(y_bound,
                                                                                                                2)))
        self.alpha = alpha
        self.t = 0

    def update(self, x, y):
        self.t += 1
        self.xtx.update(x.transpose().multiply(x))
        self.xty.update(x.transpose().multiply(y))
        theta = solve_regression(self.xtx.get().add(identity(self.d).multiply_by_num(self.t * self.alpha)),
                                 self.xty.get())
        return theta
