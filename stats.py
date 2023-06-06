from scipy.stats import norm
import numpy as np


def gamma1_func(a, b, q):
    return norm.cdf(b + (b + norm.ppf(1 - q)) / (1 - a * (b + norm.ppf(1 - q))))


def gamma2_func(a, b, q):
    return norm.cdf(b + (b + norm.ppf(q)) / (1 - a * (b + norm.ppf(q))))


def BCa_interval_macro_metric(X, func, B=1000, q=0.975):
    """
    from: https://github.com/andreped/adverse-events/blob/85cb8c59e6f3f86fdc52f985d17ba846ce0f5474/python/utils/stats.py
    :param X:
    :param func:
    :param B:
    :param q:
    :return:
    """
    X = np.array(X)
    theta_hat = func(X)
    print(theta_hat)

    N = len(X)
    order = np.array(range(N))
    order_boot = np.random.choice(order, size=(B, N), replace=True)
    X_boot = X[order_boot]

    # bootstrap
    theta_hat_boot = np.array([func(X_boot[i]) for i in range(X_boot.shape[0])])

    # 1) find jackknife estimates
    tmp = np.transpose(np.reshape(np.repeat(order, repeats=len(order)), (len(order), len(order))))  # make NxN matrix
    tmp_mat = tmp[~np.eye(tmp.shape[0], dtype=bool)].reshape(tmp.shape[0], -1)
    X_tmp_mat = X[tmp_mat]

    jk_theta = np.array([func(X_tmp_mat[i]) for i in range(X_tmp_mat.shape[0])])
    phi_jk = np.mean(jk_theta) - jk_theta  # jackknife estimates

    # 2) Find a
    a = 1 / 6 * np.sum(phi_jk ** 3) / np.sum(phi_jk ** 2) ** (3 / 2)

    # 3) Find b
    b = norm.ppf(np.sum(theta_hat_boot < theta_hat) / B)  # inverse standard normal

    # 4) Find gamma values -> limits

    # 5) get confidence interval of BCa
    CI_BCa = np.percentile(theta_hat_boot, [100 * gamma1_func(a, b, q), 100 * gamma2_func(a, b, q)])

    return CI_BCa, theta_hat_boot
