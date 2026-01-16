import math

from src.judge.Judge import evaluate
from src.test.gen_synthetic_datasets import *
from src.test.reader import read_data_file

# d = 15
# T = 200000
# raw_X, raw_y = generate_dp_hard_instance(T, d)
# eps = 1

# delta = 1.0 / T
#
# judge = Judge()
# judge.evaluate(raw_X, raw_y, T, d, eps, delta)


# T, d, raw_X, raw_y = read_data_file("files/data/Twitter.data", y_column=-1)
T = 100000
d = 15
# raw_X, raw_y = generator_2(T, d)
raw_X, raw_y = generate_dp_hard_instance(T, d)
eps = 20.0
delta = 1.0 / 100.0
evaluate(raw_X, raw_y, T, d, eps, delta, "Twitter_eps_20_delta_0.01", snapshot_interval=min(1000, math.floor(T/100.0)))

