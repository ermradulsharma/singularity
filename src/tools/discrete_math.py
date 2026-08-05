import math

def nCr(n: int, r: int) -> int:
    """Combinatorics: Computes n choose r"""
    if r > n or r < 0:
        return 0
    return math.factorial(n) // (math.factorial(r) * math.factorial(n - r))

def is_eulerian_circuit(graph_degrees: dict) -> bool:
    """Graph Theory: Checks if a graph has an Eulerian circuit"""
    for degree in graph_degrees.values():
        if degree % 2 != 0:
            return False
    return True
