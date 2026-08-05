def carnot_efficiency(t_hot: float, t_cold: float) -> float:
    """Thermodynamics: Carnot Engine Efficiency"""
    if t_hot <= 0 or t_cold <= 0:
        raise ValueError("Temperatures must be in Kelvin (> 0)")
    return 1 - (t_cold / t_hot)

def reynolds_number(density: float, velocity: float, diameter: float, viscosity: float) -> float:
    """Fluid Mechanics: Calculates Reynolds Number for fluid flow"""
    return (density * velocity * diameter) / viscosity

def gear_ratio(teeth_driven: int, teeth_driver: int) -> float:
    """Theory of Machines: Calculates Gear Ratio"""
    return teeth_driven / teeth_driver
