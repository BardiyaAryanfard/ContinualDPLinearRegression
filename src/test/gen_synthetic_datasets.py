import numpy as np


def generate_synthetic_dataset(d, T, noise_std=0.01):
    x_star = np.random.randn(d)
    print("theta_star: ", x_star.tolist())

    g_t = np.random.randn(T, d)
    eta = np.random.normal(0, noise_std, size=T)

    y_t = np.dot(g_t, x_star) + eta

    return g_t, y_t


def generate_near_low_rank(n, d, epsi=1e-3, noise_std=0.1, seed=None):
    if seed is not None:
        np.random.seed(seed)

    Z = np.random.randn(n, d)
    Sigma_sqrt = np.diag([1.0] * (d - 1) + [np.sqrt(epsi)])
    X = Z @ Sigma_sqrt

    theta_star = np.zeros(d)
    theta_star[-1] = 1.0

    y = X @ theta_star + noise_std * np.random.randn(n)

    return X, y


def generate_dp_hard_instance(
    n,
    d,
    k=None,
    x_bound=200.0,
    y_noise=0.05,
    seed=None
):
    if seed is not None:
        np.random.seed(seed)

    if k is None:
        k = max(2, d // 5)

    eps = 1.0 / np.sqrt(n)

    Q, _ = np.linalg.qr(np.random.randn(d, k))
    U = Q

    v = np.random.randn(d)
    v -= U @ (U.T @ v)
    theta_star = v / np.linalg.norm(v)

    X = np.zeros((n, d))
    y = np.zeros(n)

    for i in range(n):
        z = np.random.randn(k)
        w = np.random.randn(d)

        x = U @ z + eps * w


        norm = np.linalg.norm(x)
        if norm > 0:
            x = x / norm * x_bound

        X[i] = x
        y[i] = x @ theta_star + y_noise * np.random.randn()

    return X, y


def generator_2(n, d,
                                     X_bound=200.0,
                                     y_bound=20.0,
                                     noise_std=0.5,
                                     seed=0):

    rng = np.random.default_rng(seed)

    eigvals = np.ones(d)
    eigvals[0] = 1e-3
    eigvals[1] = 1e-2
    eigvals[2] = 1e-1

    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))

    theta_star = np.zeros(d)
    theta_star[0] = 5.0

    Z = rng.normal(size=(n, d))
    X = Z @ np.diag(np.sqrt(eigvals)) @ Q.T

    norms = np.linalg.norm(X, axis=1)
    scale = np.minimum(1.0, X_bound / (norms + 1e-12))
    X = X * scale[:, None]

    noise = rng.normal(scale=noise_std, size=n)
    y = X @ theta_star + noise

    y = np.clip(y, -y_bound, y_bound)

    return X, y


def generator_3(n, d, seed=0):
    rng = np.random.default_rng(seed)

    X_bound = 200.0
    y_bound = 20.0

    signal_strength = 0.15
    orth_noise = 0.02
    label_noise = 0.5

    theta_star = np.zeros(d)
    theta_star[0] = 5.0
    theta_star = theta_star.reshape(-1, 1)

    v = np.zeros(d)
    v[0] = 1.0
    v = v / np.linalg.norm(v)

    X = np.zeros((n, d))
    y = np.zeros((n, 1))

    for t in range(n):
        a_t = rng.normal(0, signal_strength)

        z = rng.normal(0, orth_noise, size=d)
        z = z - np.dot(z, v) * v

        x_t = a_t * v + z

        norm_x = np.linalg.norm(x_t)
        if norm_x > X_bound:
            x_t = x_t / norm_x * X_bound

        y_t = float(x_t @ theta_star) + rng.normal(0, label_noise)

        y_t = np.clip(y_t, -y_bound, y_bound)

        X[t] = x_t
        y[t] = y_t

    return X, y.flatten()






def generator_7(n, d, seed=0):
    rng = np.random.default_rng(seed)

    X_bound = 5.0
    y_bound = 20.0

    cov = 0.2
    Sigma = (1 - cov) * np.eye(d) + cov * np.ones((d, d))

    X = rng.multivariate_normal(mean=np.zeros(d), cov=Sigma, size=n)
    X = np.clip(X, -X_bound, X_bound)

    theta_star = np.zeros((d, 1))
    active_indices = rng.choice(d, size=min(5, d), replace=False)
    theta_star[active_indices, 0] = rng.uniform(2.0, 4.0, size=len(active_indices))

    y = X @ theta_star + rng.normal(0, 0.5, size=(n, 1))
    y = np.clip(y, -y_bound, y_bound)

    return X, y.flatten()




