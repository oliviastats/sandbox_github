import numpy as np

def create_bernoulli_matrix(dim_n: int, dim_m: int, p: float):
    bernoulli_matrix = np.empty((dim_n, dim_m))
    for n in range(dim_n):
        for m in range(dim_m):
            value = 1 if np.random.random() <= p else 0
            bernoulli_matrix[n][m] = value
    return bernoulli_matrix
