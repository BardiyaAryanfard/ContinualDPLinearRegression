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

        # var-rho in e1_ub is set to 1/3n
        e1_ub = 2.0 * (math.sqrt(self.d) + math.sqrt(math.log2(self.n/self.rho))) * self.X_bound * self.X_bound * math.sqrt(
            32.0 * math.log(3.75 / self.delta)) / (self.eps / 3.0) * math.log2(4.0 * self.n)
        alpha = self.lambda_min_SVT.get_alpha()
        U = e1_ub + alpha

        lam = self.c_lambda * max(0.0, U - self.current_lambda_min_tilde)

        if self.t == 0:
            print("e1_ub:", e1_ub, " alpha: ", alpha)
        if self.t % 100000 == 99999:
            print("_______________t=", self.t + 1,"________________")
            print("real_lambda_min: ", real_lambda_min)
            print("lambda_min_tilde: ", self.current_lambda_min_tilde)
            print("lambda: ", lam)
            print("SVT alpha: ", self.lambda_min_SVT.get_alpha())
            print("e1_ub: ", e1_ub)
            E1 = self.xtx.get().subtract(self.xtx.get_real_sum())
            print("||E1||_2: ", E1.spectral_norm())
            print("lambda_min(E1): ", E1.min_eigenvalue())
            A = self.xtx.get().add(identity(self.d).multiply_by_num(lam))
            A_inverse = self.xtx.get().add(identity(self.d).multiply_by_num(lam)).inverse()
            print("xtxhat: ", self.xtx.get().to_array())
            print("xtxhat^-1: ", self.xtx.get().inverse().to_array())
            print("xtxhat+lam*I: ", A.to_array())
            print("(xtxhat+lam*I)^-1: ", A_inverse.to_array())
            print("product ", A.multiply(A_inverse).to_array())
            print("xtyhat: ", self.xty.get().to_array())

        self.t += 1
        return solve_regression(self.xtx.get().add(identity(self.d).multiply_by_num(lam)), self.xty.get())
