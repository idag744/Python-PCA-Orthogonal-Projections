import numpy as np

def projection_matrix_1d(b: np.ndarray) -> np.ndarray:
    """
    Compute the projection matrix onto the 1D subspace spanned by vector b.
    """
    b = np.asarray(b, dtype=float)
    return np.outer(b, b) / np.dot(b, b)


def project_1d(x: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Project vector x onto the 1D subspace spanned by vector b.
    """
    x, b = np.asarray(x, float), np.asarray(b, float)
    coeff = np.dot(b, x) / np.dot(b, b)
    return coeff * b


def projection_matrix_general(B: np.ndarray) -> np.ndarray:
    """
    Compute the projection matrix onto the subspace spanned by columns of B.
    """
    B = np.asarray(B, dtype=float)
    BTB_inv = np.linalg.inv(B.T @ B)
    return B @ BTB_inv @ B.T


def project_general(x: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Project vector x onto the subspace spanned by columns of B.
    """
    x = np.asarray(x, float).reshape(-1, 1)
    B = np.asarray(B, float)
    BTB_inv = np.linalg.inv(B.T @ B)
    return B @ BTB_inv @ B.T @ x

