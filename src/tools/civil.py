def bending_stress(moment: float, y: float, moment_of_inertia: float) -> float:
    """Strength of Materials (SOM): Bending Stress Equation (sigma = M*y/I)"""
    return (moment * y) / moment_of_inertia

def terzaghi_bearing_capacity(c: float, Nc: float, q: float, Nq: float, gamma: float, B: float, Ngamma: float) -> float:
    """Geotechnical Engineering: Terzaghi's Bearing Capacity Equation for continuous footings"""
    return (c * Nc) + (q * Nq) + (0.5 * gamma * B * Ngamma)

def stopping_sight_distance(velocity_kmh: float, reaction_time: float, friction_coeff: float) -> float:
    """Transportation Engineering: Calculates Stopping Sight Distance (SSD)"""
    v_ms = velocity_kmh * (5 / 18)
    lag_distance = v_ms * reaction_time
    braking_distance = (v_ms ** 2) / (2 * 9.81 * friction_coeff)
    return lag_distance + braking_distance
