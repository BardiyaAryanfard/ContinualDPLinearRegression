import math

from src.util.BinaryTree import BCMOpt
from src.util.Matrix import *
from src.util.SVT import SVT


def gaussian_noise_variance(eps, delta, sensitivity_squared):
    return 2.0 * sensitivity_squared * math.log(1.25 / delta) / (eps * eps)


class ContinualAdaSSP:
    def __init__(self, n, d, eps, delta, X_bound, y_bound):
        self.n = n
        self.d = d
        self.eps = eps
        self.delta = delta
        self.X_bound = X_bound
        self.y_bound = y_bound
        self.xtx = BCMOpt(n, eps / 3.0, d, d, gaussian_noise_variance(eps / 3.0, delta / 3.0,
                                                                      4.0 * math.log2(n) * math.pow(X_bound, 4)))
        self.xty = BCMOpt(n, eps / 3.0, d, 1, gaussian_noise_variance(eps / 3.0, delta / 3.0,
                                                                      4 * math.log2(n) * math.pow(X_bound,
                                                                                                  2) * math.pow(y_bound,
                                                                                                                2)))
        self.lambda_min_SVT = SVT(eps / 3.0, delta / 3.0, n, 4.0 * X_bound * X_bound, math.ceil(math.log2(n)))
        self.current_lambda_min_tilde = 0
        self.SVT_threshold = max(1.0, 4.0 * self.X_bound * self.X_bound)

    def update(self, x, y):
        self.xtx.update(x.transpose().multiply(x))
        self.xty.update(x.transpose().multiply(y))
        if self.lambda_min_SVT.update(self.xtx.get_real_sum().get_min_eigenvalue(), self.SVT_threshold) is True:
            self.current_lambda_min_tilde = self.SVT_threshold
            self.SVT_threshold *= 2.0
        # var-rho in e1_ub is set to 1/3n
        e1_ub = 24.0 * self.X_bound * self.X_bound * math.log2(self.n) * math.sqrt(
            self.d * math.log(2.5 / self.delta) * math.log(6.0 * self.d * self.d * self.n)) / self.eps
        alpha = self.lambda_min_SVT.get_alpha()
        lam = max(0.0, e1_ub + alpha - self.current_lambda_min_tilde)
        return (self.xtx.get().add(identity(self.d).multiply_by_num(lam))).inverse().multiply(self.xty.get())
