import math

def calculate_impedance(resistance: float, inductive_reactance: float, capacitive_reactance: float) -> float:
    """Electrical Engineering: RLC Circuit Impedance Calculation"""
    return math.sqrt(resistance**2 + (inductive_reactance - capacitive_reactance)**2)

def ohm_law(voltage=None, current=None, resistance=None):
    """Basic Electronics: Ohm's Law Solver. Pass exactly 2 arguments."""
    if voltage is None and current is not None and resistance is not None:
        return current * resistance
    if current is None and voltage is not None and resistance is not None:
        return voltage / resistance
    if resistance is None and voltage is not None and current is not None:
        return voltage / current
    return None
