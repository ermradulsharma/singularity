def laplace_transform_exp(a: float, s: float):
    """Engineering Mathematics: Laplace Transform of e^(at), L{e^(at)} = 1 / (s - a)"""
    if s > a:
        return 1 / (s - a)
    return "Undefined (requires s > a)"

def determinant_2x2(matrix: list) -> float:
    """Linear Algebra: Calculates determinant of a 2x2 matrix"""
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

def cross_product_3d(v1: list, v2: list) -> list:
    """Vector Calculus: Cross product of two 3D vectors"""
    return [
        v1[1]*v2[2] - v1[2]*v2[1],
        v1[2]*v2[0] - v1[0]*v2[2],
        v1[0]*v2[1] - v1[1]*v2[0]
    ]
