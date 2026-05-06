import math
import time


from src.algorithms.AdaSSP import *
from src.judge.logger import OutputLogger
from src.util.IO import *
from src.util.Matrix import Matrix, solve_regression
from src.algorithms.Continual_AdaSSP import ContinualAdaSSP
from src.algorithms.PQFTL import PQFTL


def F(X, y, theta):
    diff = X.multiply(theta).subtract(y)
    return 0.5 * math.pow(np.linalg.norm(diff.to_array(), ord=2), 2)


def evaluate_continual(eps, delta, exp_name, data_set, snapshot_interval=1000, output_path="files/logs/"):
    start_time = time.time()

    T = data_set.n
    d = data_set.d
    X_bound = data_set.X_bound
    y_bound = data_set.y_bound


    c_lambdas = [1.0/16.0, 1.0]
    c_lambdas_str = ["1/16", "1"]

    Ms = [ContinualAdaSSP(T, d, eps, delta, X_bound, y_bound, c_lambda=c_lambda) for c_lambda in c_lambdas]

    pqftl = PQFTL(T, d, eps, delta, 1.0, X_bound, y_bound)

    output_logger = OutputLogger(curve_names=[("AdaSSP"+c) for c in c_lambdas_str] + ["Batch", "PQFTL", "non-private", "trivial"], exp_name=exp_name)


    print("preparing done, time used:", time.time() - start_time)
    prev_time = time.time()

    eps_comp = eps / (2.0 * math.sqrt(2.0 * T * math.log(2.0 / delta)))
    delta_comp = delta / (2.0 * T)
    X = Matrix(T, d)
    y = Matrix(T, 1)

    for i in range(T):
        plain_x, plain_y = data_set.next_data_point()
        x = Matrix(1, d, plain_x)
        y_i = Matrix(1, 1, [[plain_y]])
        X.set_row(i, x)
        y.set_row(i, y_i)
        if x.transpose().multiply(x).data[0][0] is None or np.isnan(x.transpose().multiply(x).data[0][0]):
            print("NaN detected in x at time", i + 1)
            print("x:", x.to_array())
            continue
        theta_Ms = [Ms[j].update(x, y_i) for j in range(len(c_lambdas))]
        theta_pqftl = pqftl.update(x, y_i)
        if i > 10 and i % snapshot_interval == snapshot_interval - 1:
            if i % 1000 == 999:
                print("-----------------t=", i + 1, "-----------------")
                print("overall time used:", time.time() - start_time)
                print("time used for last", snapshot_interval, "iterations:", time.time() - prev_time)
            prev_time = time.time()
            cur_x = X.first_rows(i + 1)
            cur_y = y.first_rows(i + 1)
            theta_star = solve_regression(cur_x, cur_y)
            F_0 = F(cur_x, cur_y, Matrix(d, 1, np.zeros((d, 1))))
            F_pqftl = F(cur_x, cur_y, theta_pqftl)
            F_batch = F(cur_x, cur_y, solve_new(i+1, d, eps_comp, delta_comp, X_bound, y_bound, cur_x, cur_y))
            F_theta_star = F(cur_x, cur_y, theta_star)
            output_logger.add_point(i + 1, [F(cur_x,cur_y, theta_Ms[j]) for j in range(len(c_lambdas))] +
                                    [F_batch, F_pqftl, F_theta_star, F_0])

    output_logger.print_to_file(exp_name, output_path)



