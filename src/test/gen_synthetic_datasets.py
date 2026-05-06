import numpy as np


def generator_heuristic(d, T, condition_number, noise_std, X_bound=1.0, y_bound=None, seed=42):
    if y_bound is None:
        y_bound = X_bound
    rng = np.random.default_rng(seed)

    if d == 1:
        stds = np.array([1.0])
    else:
        log_ratio = np.log(condition_number) / (d - 1)
        stds = np.exp(-0.5 * log_ratio * np.arange(d))
    stds *= X_bound / (np.sqrt(np.sum(stds ** 2)) + 1e-15) * np.sqrt(d) * 0.3

    X = rng.standard_normal((T, d)) * stds[None, :]

    theta_star = rng.standard_normal(d)
    theta_star /= np.linalg.norm(theta_star)

    y = X @ theta_star + rng.normal(0, noise_std, size=T)

    X = np.clip(X, -X_bound, X_bound)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X * np.minimum(1.0, X_bound / norms)
    y = np.clip(y, -y_bound, y_bound)

    perm = rng.permutation(T)
    return len(y), d, X[perm], y[perm], X_bound, y_bound

# BENCHMARK = {
#     "S21_d05_cond10_lownoise":     (T,  5,   10,  0.03),
#     "S22_d05_cond100_mednoise":    (T,  5,  100,  0.15),
#     "S23_d05_cond1000_highnoise":  (T,  5, 1000,  0.30),
#     "S24_d10_cond10_lownoise":     (T,  10,   10,  0.03),
#     "S25_d10_cond100_mednoise":    (T,  10,  100,  0.15),
#     "S26_d10_cond1000_highnoise":  (T,  10, 1000,  0.30),
#     "S27_d15_cond10_lownoise":     (T,  15,   10,  0.03),
#     "S28_d15_cond100_mednoise":    (T,  15,  100,  0.15),
#     "S29_d15_cond1000_highnoise":  (T,  15, 1000,  0.30),
#     "S30_d30_cond10_lownoise":     (T,  30,   10,  0.03),
#     "S31_d30_cond100_mednoise":    (T,  30,  100,  0.15),
#     "S32_d30_cond1000_highnoise":  (T,  30, 1000,  0.30),
# }

