import math
import time

from src.judge.logger import OutputLogger
from src.util.IO import *
from src.util.Matrix import Matrix
from src.algorithms.Continual_AdaSSP import ContinualAdaSSP
from src.algorithms.PQFTL import PQFTL


def F(X, y, theta):
    diff = X.multiply(theta).subtract(y)
    return 0.5 * math.pow(np.linalg.norm(diff.to_array(), 2), 2)


def evaluate(raw_X, raw_y, T, d, eps, delta, exp_name, snapshot_interval):
    start_time = time.time()
    print(raw_X)
    print(raw_y)
    print(raw_X.shape)
    print(raw_y.shape)
    X_bound = 0
    for x in raw_X:
        X_bound = max(X_bound, np.linalg.norm(x, ord=2))
    y_bound = 0
    for y_i in raw_y:
        y_bound = max(y_bound, abs(y_i))

    # for i in range(T):
    #     for j in range(d):
    #         raw_X[i][j] = raw_X[i][j] / (X_bound * 1.0)
    #     raw_y[i] = raw_y[i] / (y_bound * 1.0)

    print("X_bound: ", X_bound)
    print("y_bound: ", y_bound)

    X_bound = 0
    for x in raw_X:
        X_bound = max(X_bound, np.linalg.norm(x, ord=2))
    y_bound = 0
    for y_i in raw_y:
        y_bound = max(y_bound, abs(y_i))

    M = ContinualAdaSSP(T, d, eps, delta, X_bound, y_bound)
    pqftl = PQFTL(T, d, eps, delta, 1.0, X_bound, y_bound)
    print(math.sqrt(M.xtx.root.noise_variance))

    output_logger = OutputLogger(curve_names=["AdaSSP", "PQFTL", "theta*", "zero"], exp_name=exp_name)

    X = Matrix(T, d, raw_X)
    y = Matrix(T, 1, raw_y.reshape(-1, 1))

    print("preparing done, time used:", time.time() - start_time)
    prev_time = time.time()

    for i in range(T):
        x = Matrix(1, d, raw_X[i].reshape(1, -1))
        y_i = Matrix(1, 1, [[raw_y[i]]])
        theta = M.update(x, y_i)
        theta_pqftl = pqftl.update(x, y_i)
        if i % snapshot_interval == snapshot_interval - 1:
            print("-----------------t=", i + 1, "-----------------")
            print("overall time used:", time.time() - start_time)
            print("time used for last", snapshot_interval, "iterations:", time.time() - prev_time)
            prev_time = time.time()
            cur_x = X.first_rows(i + 1)
            cur_y = y.first_rows(i + 1)
            # print(cur_x.to_array())
            theta_star = Matrix(d, 1, np.linalg.pinv(cur_x.to_array()) @ cur_y.to_array())
            print("theta_AdaSSP=", theta.to_array())
            print("theta_star=", theta_star.to_array())
            print("F(0)=", F_0 := F(cur_x, cur_y, Matrix(d, 1, np.zeros((d, 1)))))
            print("F(theta_AdaSSP)=", F_adassp := F(cur_x, cur_y, theta))
            print("F(theta_pqftl)=", F_pqftl := F(cur_x, cur_y, theta_pqftl))
            print("F_min=", F_theta_star := F(cur_x, cur_y, theta_star))
            output_logger.add_point(i + 1, [F_adassp, F_pqftl, F_theta_star, F_0])
        # if i % (50 * snapshot_interval) == 50 * snapshot_interval - 1:
        #     output_logger.show()
        #     output_logger.save()
        #     output_logger.print_to_file(exp_name)

    output_logger.show()
    output_logger.save()
    output_logger.print_to_file(exp_name)
