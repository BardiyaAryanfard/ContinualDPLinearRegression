import math
import time


from src.algorithms.AdaSSP import *
from src.judge.logger import OutputLogger
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
    Ms_pretime = []
    Ms = []
    for c_lambda in c_lambdas:
        t_prev = time.time()
        Ms.append(ContinualAdaSSP(T, d, eps, delta, X_bound, y_bound, c_lambda=c_lambda))
        Ms_pretime.append(time.time() - t_prev)
    t_prev = time.time()
    pqftl = PQFTL(T, d, eps, delta, 1.0, X_bound, y_bound)
    pqftl_pretime = time.time() - t_prev

    output_logger = OutputLogger(curve_names=[("AdaSSP"+c) for c in c_lambdas_str] + ["Batch", "PQFTL", "non-private", "trivial"], exp_name=exp_name)

    theta_star_time = 0.0
    batch_time = 0.0
    pqftl_time = 0.0
    Ms_time = [0.0 for _ in c_lambdas]

    eps_comp = eps / (2.0 * math.sqrt(2.0 * T * math.log(2.0 / delta)))
    delta_comp = delta / (2.0 * T)
    X = Matrix(T, d)
    y = Matrix(T, 1)

    for i in range(T):
        plain_x, plain_y = data_set.next_data_point()
        x = Matrix(1, d, plain_x)
        y_i = Matrix(1, 1, [[plain_y]])
        t_prev = time.time()
        X.set_row(i, x)
        y.set_row(i, y_i)
        xtx = x.transpose().multiply(x)
        if xtx.data[0][0] is None or np.isnan(xtx.data[0][0]):
            continue
        t_reading = time.time() - t_prev
        theta_Ms = []
        for j in range(len(c_lambdas)):
            prev_time = time.time()
            theta_Ms.append(Ms[j].update(x, y_i))
            Ms_time[j] += time.time() - prev_time
        prev_time = time.time()
        theta_pqftl = pqftl.update(x, y_i)
        pqftl_time += time.time() - prev_time
        if i > 10 and i % snapshot_interval == snapshot_interval - 1:
            cur_x = X.first_rows(i + 1)
            cur_y = y.first_rows(i + 1)

            prev_time = time.time()
            theta_star = solve_regression(cur_x, cur_y)
            theta_star_time += time.time() - prev_time

            prev_time = time.time()
            theta_batch = solve_new(i + 1, d, eps_comp, delta_comp, X_bound, y_bound, cur_x, cur_y)
            batch_time += time.time() - prev_time

            F_0 = F(cur_x, cur_y, Matrix(d, 1, np.zeros((d, 1))))
            F_pqftl = F(cur_x, cur_y, theta_pqftl)
            F_batch = F(cur_x, cur_y, theta_batch)
            F_theta_star = F(cur_x, cur_y, theta_star)

            output_logger.add_point(i + 1, [F(cur_x,cur_y, theta_Ms[j]) for j in range(len(c_lambdas))] +
                                    [F_batch, F_pqftl, F_theta_star, F_0])
    output_logger.set_running_times([Ms_time[j] for j in range(len(c_lambdas))] + [batch_time, pqftl_time, theta_star_time, 0.0])
    output_logger.set_preprocessing_times([Ms_pretime[j] for j in range(len(c_lambdas))] + [0.0, pqftl_pretime, 0.0, 0.0])
    output_logger.set_number_of_time_steps([T for _ in range(len(c_lambdas))] + [T/snapshot_interval, T, T/snapshot_interval, T/snapshot_interval])
    output_logger.print_to_file(exp_name, output_path)



