from .norm_cols import normalize_columns, normalize_power_columns
import numpy as np
from scipy.spatial.distance import cdist


def _fcm_step(data, u_old, c, m, metric):
    """
    """
    # Normalizing, then eliminating any potential zero values.
    u_old = normalize_columns(u_old)
    u_old = np.fmax(u_old, np.finfo(np.float64).eps)

    um = u_old ** m

    # Calculate cluster centers
    data = data.T
    cntr = um.dot(data) / np.atleast_2d(um.sum(axis=1)).T

    d = _distance(data, cntr, metric)
    d = np.fmax(d, np.finfo(np.float64).eps)

    jm = (um * d ** 2).sum()

    u = normalize_power_columns(d, - 2. / (m - 1))

    return cntr, u, jm, d


def _distance(data, centers, metric='euclidean'):
    """
    scipy.spatial.distance.cdist
    """
    return cdist(data, centers, metric=metric).T


def _fp_coeff(u):
    """
    """
    n = u.shape[1]

    return np.trace(u.dot(u.T)) / float(n)


# точка входа в кластеризацию
def fcm(data, c, m, error, maxiter,
        metric='euclidean',
        init=None, seed=None):
    """
    """
    # Setup u0
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


def _fcm_step_predict(test_data, cntr, u_old, c, m, metric):
    """
    """
    u_old = normalize_columns(u_old)
    u_old = np.fmax(u_old, np.finfo(np.float64).eps)

    um = u_old ** m
    test_data = test_data.T

    d = _distance(test_data, cntr, metric)
    d = np.fmax(d, np.finfo(np.float64).eps)

    jm = (um * d ** 2).sum()

    u = normalize_power_columns(d, - 2. / (m - 1))

    return u, jm, d


def fcm_predict(test_data, cntr_trained, m, error, maxiter,
                metric='euclidean',
                init=None,
                seed=None):
    """
    """
    c = cntr_trained.shape[0]

    # Setup u0
    if init is None:
        if seed is not None:
            np.random.seed(seed=seed)
        n = test_data.shape[1]
        u0 = np.random.rand(c, n)
        u0 = normalize_columns(u0)
        init = u0.copy()
    u0 = init
    u = np.fmax(u0, np.finfo(np.float64).eps)

    # Initialize loop parameters
    jm = np.zeros(0)
    p = 0

    # Main cmeans loop
    while p < maxiter:
        u2 = u.copy()
        [u, Jjm, d] = _fcm_step_predict(test_data, cntr_trained, u2, c, m,
                                        metric)
        jm = np.hstack((jm, Jjm))
        p += 1

        # Stopping rule
        if np.linalg.norm(u - u2) < error:
            break

    # Final calculations
    error = np.linalg.norm(u - u2)
    fpc = _fp_coeff(u)

    return u, u0, d, jm, p, fpc
