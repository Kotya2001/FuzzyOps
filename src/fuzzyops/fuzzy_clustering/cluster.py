import numpy as np
from scipy.spatial.distance import cdist


def _fcm_step(data, u_old, c, m, metric):
    u_old = normalize_columns(u_old)
    u_old = np.fmax(u_old, np.finfo(np.float64).eps)

    um = u_old ** m

    data = data.T
    cntr = um.dot(data) / np.atleast_2d(um.sum(axis=1)).T

    d = _distance(data, cntr, metric)
    d = np.fmax(d, np.finfo(np.float64).eps)

    jm = (um * d ** 2).sum()

    u = normalize_power_columns(d, - 2. / (m - 1))

    return cntr, u, jm, d


def _distance(data, centers, metric='euclidean'):
    return cdist(data, centers, metric=metric).T


def _fp_coeff(u):
    n = u.shape[1]

    return np.trace(u.dot(u.T)) / float(n)


def fcm(data, c, m, error, maxiter,
            metric='euclidean',
            init=None, seed=None):
    if init is None:
        if seed is not None:
            np.random.seed(seed=seed)
        n = data.shape[1]
        u0 = np.random.rand(c, n)
        u0 = normalize_columns(u0)
        init = u0.copy()
    u0 = init
    u = np.fmax(u0, np.finfo(np.float64).eps)

    jm = np.zeros(0)
    p = 0

    while p < maxiter:
        u2 = u.copy()
        [cntr, u, Jjm, d] = _fcm_step(data, u2, c, m, metric)
        jm = np.hstack((jm, Jjm))
        p += 1

        if np.linalg.norm(u - u2) < error:
            break

    error = np.linalg.norm(u - u2)
    fpc = _fp_coeff(u)

    return cntr, u, u0, d, jm, p, fpc


def fcm_predict(test_data, cntr_trained, m, error, maxiter,
                metric='euclidean',
                init=None,
                seed=None):
    c = cntr_trained.shape[0]

    if init is None:
        if seed is not None:
            np.random.seed(seed=seed)
        n = test_data.shape[1]
        u0 = np.random.rand(c, n)
        u0 = normalize_columns(u0)
        init = u0.copy()
    u0 = init
    u = np.fmax(u0, np.finfo(np.float64).eps)

    jm = np.zeros(0)
    p = 0

    while p < maxiter:
        u2 = u.copy()
        [u, Jjm, d] = _fcm_step_predict(test_data, cntr_trained, u2, c, m,
                                       metric)
        jm = np.hstack((jm, Jjm))
        p += 1

        if np.linalg.norm(u - u2) < error:
            break

    error = np.linalg.norm(u - u2)
    fpc = _fp_coeff(u)

    return u, u0, d, jm, p, fpc


def _fcm_step_predict(test_data, cntr, u_old, c, m, metric):

    u_old = normalize_columns(u_old)
    u_old = np.fmax(u_old, np.finfo(np.float64).eps)

    um = u_old ** m
    test_data = test_data.T

    d = _distance(test_data, cntr, metric)
    d = np.fmax(d, np.finfo(np.float64).eps)

    jm = (um * d ** 2).sum()

    u = normalize_power_columns(d, - 2. / (m - 1))

    return u, jm, d


# нормализация колонок матрицы
def normalize_columns(columns):
    normalized_columns = columns / np.sum(columns, axis=0, keepdims=1)
    return normalized_columns


def normalize_power_columns(x, exp):
    assert np.all(x >= 0.0)

    x = x.astype(np.float64)

    x = x / np.max(x, axis=0, keepdims=True)

    x = np.fmax(x, np.finfo(x.dtype).eps)

    if exp < 0:
        x /= np.min(x, axis=0, keepdims=True)
        x = x ** exp
    else:
        x = x ** exp

    result = normalize_columns(x)

    return result
