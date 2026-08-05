import scipy.constants as const
import math

def escape_velocity(mass: float, radius: float) -> float:
    """Calculate escape velocity of a planet. v = sqrt(2GM/R)"""
    return math.sqrt(2 * const.G * mass / radius)

def orbital_period(mass: float, radius: float, altitude: float) -> float:
    """Calculate orbital period of a satellite using Kepler's Third Law."""
    a = radius + altitude
    return 2 * math.pi * math.sqrt((a**3) / (const.G * mass))
