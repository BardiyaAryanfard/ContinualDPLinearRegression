import math

import numpy as np

from src.util import Matrix
from src.util.Matrix import *


def solve_new(n, d, eps, delta, X_bound, y_bound, X, y):
    rho = 0.1
    var_xtx = 2.0 * math.log(3.75 / delta) / math.pow(eps / 3.0, 2) * math.pow(X_bound, 4)
    var_xty = 2.0 * math.log(3.75 / delta) / math.pow(eps / 3.0, 2) * math.pow(X_bound, 2) * math.pow(y_bound, 2)
    var_lambda_min = 2.0 * math.log(3.75 / delta) / math.pow(eps / 3.0, 2) * math.pow(X_bound, 4)
    U = math.sqrt(var_lambda_min) * (math.sqrt(d) + math.sqrt(1.0 / rho)) * 4.0
    XTX_hat = X.transpose().multiply(X).add(generate_symmetric_gaussian_matrix(d, var_xtx))
    XTY_hat = X.transpose().multiply(y).add(generate_gaussian_matrix(d, 1, var_xty))
    lambda_min_tilde = (X.transpose().multiply(X)).min_eigenvalue() + np.random.normal(0,
                                                                                       var_lambda_min) - math.sqrt(
        var_lambda_min) * math.sqrt(2.0 * math.log(3.75 / delta))
    lam = max(0.0, U - lambda_min_tilde)
    theta_hat = XTX_hat.add(identity(d).multiply_by_num(lam)).inverse().multiply(XTY_hat)
    return theta_hat


def solve_old(n, d, eps, delta, X_bound, y_bound, X, y):
    rho = 0.1
    var_xtx = math.log(6.0 / delta) / math.pow(eps / 3.0, 2) * math.pow(X_bound, 4)
    var_xty = math.log(6.0 / delta) / math.pow(eps / 3.0, 2) * math.pow(X_bound, 2) * math.pow(y_bound, 2)
    var_lambda_min = math.log(6.0 / delta) / math.pow(eps / 3.0, 2) * math.pow(X_bound, 4)
    U = math.sqrt(var_lambda_min) * math.sqrt(d * math.log(2.0 * d * d / rho))
    XTX_hat = X.transpose().multiply(X).add(generate_symmetric_gaussian_matrix(d, var_xtx))
    XTY_hat = X.transpose().multiply(y).add(generate_gaussian_matrix(d, 1, var_xty))
    lambda_min_tilde = (X.transpose().multiply(X)).min_eigenvalue() + np.random.normal(0,
                                                                                       var_lambda_min) - 2.0 * math.sqrt(
        var_lambda_min) * math.sqrt(math.log(6.0 / delta))
    lam = max(0.0, U - lambda_min_tilde)
    theta_hat = XTX_hat.add(identity(d).multiply_by_num(lam)).inverse().multiply(XTY_hat)
    return theta_hat


