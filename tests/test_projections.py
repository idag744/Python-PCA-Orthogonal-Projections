import numpy as np
from numpy.testing import assert_allclose
from projections.core import projection_matrix_1d, project_1d, projection_matrix_general, project_general

def test_projection_1d():
    b = np.array([1, 2, 2])
    P = projection_matrix_1d(b)
    assert_allclose(P, np.array([[1,2,2],[2,4,4],[2,4,4]])/9)
    x = np.ones(3)
    y = project_1d(x, b)
    assert_allclose(y, np.array([5,10,10])/9)

def test_projection_nd():
    B = np.array([[1,0],[1,1],[1,2]])
    P = projection_matrix_general(B)
    assert_allclose(P, np.array([[5,2,-1],[2,2,2],[-1,2,5]])/6)
    x = np.array([6,0,0])
    y = project_general(x, B)
    assert_allclose(y.flatten(), np.array([5,2,-1]))

