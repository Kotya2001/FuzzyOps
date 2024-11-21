

import numpy as np

from fuzzyops.graphs.fuzzgraph import FuzzyGraph
from typing import List


def mle_clusterization_factors(graph: FuzzyGraph, clusters_amount: int) -> List[int]:
    """
    Выполняет кластеризацию узлов в нечетком графе с использованием метода MLE.

    Метод максимизации правдоподобия (MLE) позволяет выделить несколько кластеров
    узлов, основываясь на их взаимосвязях и значениях.

    Args:
        graph (FuzzyGraph): Экземпляр нечеткого графа, содержащего узлы для кластеризации.
        clusters_amount (int): Количество кластеров, на которое необходимо разбить узлы.

    Returns:
        List[int]: Список индексов кластеров, к которым принадлежит каждый узел графа.

    Raises:
        Exception: Исключение возникает, если:
            - `graph` не является экземпляром класса `FuzzyGraph`.
            - `clusters_amount` не является целым числом.
    """

    if not(type(graph) is FuzzyGraph):
        raise Exception('Can use only FuzzGraph')

    if not(type(clusters_amount) is int):
        raise Exception('Clusters amount could be only integer')

    N = graph.get_nodes_amount()
    D = 1
    max_iterations = 100
    tol = 1e-6

    # Инициализация средних значений mu
    np.random.seed(42)
    mu = np.zeros((clusters_amount, D))
    g = np.random.rand(N, 1)
    for k in range(clusters_amount):
        nodes_in_cluster = np.random.choice(N, int(N / clusters_amount), replace=False)
        mu[k, :] = np.mean(g[nodes_in_cluster,:], axis=0)

    # Инициализация ковариационных матриц Sigma
    Sigma = np.zeros((clusters_amount, D, D))
    for k in range(clusters_amount):
        Sigma[k, :, :] = np.eye(D)

    # Инициализация весов pi
    pi = np.ones(clusters_amount) / clusters_amount

    log_likelihoods = []
    old_log_likelihood = -np.inf
    gamma = []

    for iteration in range(max_iterations):

        # E-шаг
        log_prob = np.zeros((N, clusters_amount))
        for k in range(clusters_amount):


            diff = g - mu[k, :]
            norm_coeff = -0.5 * clusters_amount * np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(Sigma[k, :, :]))
            exponent = -0.5 * np.sum(np.matmul(diff, np.linalg.inv(Sigma[k, :, :])) * diff, axis=1)
            log_prob[:, k] = np.log(pi[k]) + norm_coeff + exponent

        log_likelihood = np.sum(np.max(log_prob, axis=1))
        log_likelihoods.append(log_likelihood)

        # M-шаг
        gamma = np.exp(log_prob - np.max(log_prob, axis=1)[:, np.newaxis])
        gamma /= np.sum(gamma, axis=1)[:, np.newaxis]

        N_k = np.sum(gamma, axis=0)
        mu = np.dot(gamma.T, g) / N_k[:, np.newaxis]
        for k in range(clusters_amount):
            X_centered = g - mu[k, :]
            Sigma[k, :, :] = np.dot(X_centered.T, gamma[:, k][:, np.newaxis] * X_centered) / N_k[k]
        pi = N_k / N

        # Проверка сходимости
        if np.abs(log_likelihood - old_log_likelihood) < tol:
            break

        old_log_likelihood = log_likelihood

    return list(np.argmax(gamma, axis=1))

