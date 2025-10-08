import math

import numpy as np

from src.algorithms.Continual_AdaSSP import ContinualAdaSSP
from src.util.IO import *
from src.util.Matrix import Matrix
from src.algorithms.PQFTL import PQFTL


def generate_synthetic_dataset(d=10, T=100000, noise_std=0.01):
    # Generate x* as a random vector
    x_star = np.random.randn(d)
    print("theta_star: ", x_star.tolist())

    # Generate g_t as samples from a multivariate Gaussian distribution
    g_t = np.random.randn(T, d)
    # Generate η as Gaussian noise with standard deviation 0.01
    eta = np.random.normal(0, noise_std, size=T)

    # Compute y_t = g_t^T x* + η
    y_t = np.dot(g_t, x_star) + eta

    return g_t, y_t


d = 20
T = 100000
raw_X, raw_y = generate_synthetic_dataset(d, T, noise_std=0.01)
# print(raw_X.shape)
# print(raw_y.shape)
X_bound = 0
for x in raw_X:
    X_bound = max(X_bound, np.linalg.norm(x, ord=2))
print("X_bound: ", X_bound)
y_bound = 0
for y_i in raw_y:
    y_bound = max(y_bound, abs(y_i))
print("y_bound: ", y_bound)

# write_input_dataset("data/test/test1.csv", X, y)

X = Matrix(T, d, raw_X)
y = Matrix(T, 1, raw_y.reshape(-1, 1))

# print(X.to_array())
# print(y.to_array())

M = ContinualAdaSSP(T, d, 1.0, 1e-5, X_bound, y_bound)
PQFTL = PQFTL(T, d, 1.0, 1e-5, 1.0, X_bound, y_bound)
print(math.sqrt(M.xtx.root.noise_variance))

for i in range(T):
    x = Matrix(1, d, raw_X[i].reshape(1, -1))
    y_i = Matrix(1, 1, [[raw_y[i]]])
    theta = M.update(x, y_i)
    theta_pqftl = PQFTL.update(x, y_i)
    cur_x = X.first_rows(i + 1)
    cur_y = y.first_rows(i + 1)
    # if i % 10000 == 0:
    #     print("xtx hat - xtx:", cur_x.transpose().multiply(cur_x).subtract(M.xtx.get()).to_array())
    #     print("xtx:", cur_x.transpose().multiply(cur_x).to_array())
    #     print("xty hat - xty:", cur_x.transpose().multiply(cur_y).subtract(M.xty.get()).to_array())
    #     print("xty:", cur_x.transpose().multiply(cur_y).to_array())
    # print("t=", i + 1, " theta: ", theta.to_array())
    if i > 73 and i % 10000 == 0:
        cur_x = X.first_rows(i + 1)
        cur_y = y.first_rows(i + 1)
        # print(cur_x.to_array())
        theta_star = (cur_x.transpose().multiply(cur_x)).inverse().multiply(cur_x.transpose().multiply(cur_y))
        print("t=", i + 1, " theta_star: ", theta_star.to_array())
        print("t=", i + 1, " theta_AdaSSP: ", theta.to_array())
        print("t=", i + 1, " theta_PQFTL: ", theta_pqftl.to_array())
        print("F(theta)=", np.linalg.norm(cur_x.multiply(theta).subtract(cur_y).to_array(), 2))
        print("F(theta_pqftl)=", np.linalg.norm(cur_x.multiply(theta_pqftl).subtract(cur_y).to_array(), 2))
        print("F_min=", np.linalg.norm(cur_x.multiply(theta_star).subtract(cur_y).to_array(), 2))
