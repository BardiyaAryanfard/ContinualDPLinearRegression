import math

from src.util.BinaryTree import BCMOpt, gaussian_noise_variance
from src.util.Matrix import *
from src.util.SVT import SVT


class ContinualAdaSSP:
    def __init__(self, n, d, eps, delta, X_bound, y_bound, c_lambda = 1.0):
        self.n = n
        self.d = d
        self.eps = eps
        self.delta = delta
        self.X_bound = X_bound
        self.y_bound = y_bound
        self.c_lambda = c_lambda
        self.rho = 0.1
        self.xtx = BCMOpt(n, eps / 3.0, d, d, gaussian_noise_variance(eps / 3.0, delta / 3.0,
                                                                      4.0 * math.log2(n) * math.pow(X_bound, 4)))
        self.xty = BCMOpt(n, eps / 3.0, d, 1, gaussian_noise_variance(eps / 3.0, delta / 3.0,
                                                                      4 * math.log2(n) * math.pow(X_bound,
                                                                                                  2) * math.pow(y_bound,
                                                                                                                2)))
        self.lambda_min_SVT = SVT(eps / 3.0, delta / 3.0, n, 4.0 * X_bound * X_bound, math.ceil(math.log2(n)))
        self.current_lambda_min_tilde = 0
        self.SVT_threshold = max(1.0, 4.0 * self.X_bound * self.X_bound)
        self.t = 0

    def update(self, x, y):
        self.xtx.update(x.transpose().multiply(x))
        self.xty.update(x.transpose().multiply(y))
        real_lambda_min = max(0.0, self.xtx.get_real_sum().min_eigenvalue())
        if self.lambda_min_SVT.update(real_lambda_min, self.SVT_threshold) is True:
            self.current_lambda_min_tilde = self.SVT_threshold
            self.SVT_threshold *= 2.0
            self.SVT_threshold = min(self.SVT_threshold, 4.0 * self.X_bound * self.X_bound * (self.t + 1))

        e1_ub = 2.0 * (math.sqrt(self.d) + math.sqrt(math.log2(self.n/self.rho))) * self.X_bound * self.X_bound * math.sqrt(
            32.0 * math.log(3.75 / self.delta)) / (self.eps / 3.0) * math.log2(4.0 * self.n)
        alpha = self.lambda_min_SVT.get_alpha()
        U = e1_ub + alpha

        lam = self.c_lambda * max(0.0, U - self.current_lambda_min_tilde)

        if self.t == 0:
            print("e1_ub:", e1_ub, " alpha: ", alpha)

        self.t += 1
        return solve_regression(self.xtx.get().add(identity(self.d).multiply_by_num(lam)), self.xty.get())
