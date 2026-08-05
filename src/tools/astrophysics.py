import math

class Astro_Mechanic:
    """Agent specialized in Astrophysics, Orbital Mechanics, and Relativity calculations."""
    
    def calculate_hohmann_transfer(self, r1: float, r2: float, mass_central_body: float):
        """
        Calculates the delta-v required for a Hohmann transfer orbit.
        r1: Radius of initial circular orbit (meters)
        r2: Radius of target circular orbit (meters)
        mass_central_body: Mass of the body being orbited (kg)
        """
        G = 6.67430e-11 # Gravitational constant
        mu = G * mass_central_body
        
        # Velocity in initial orbit
        v1 = math.sqrt(mu / r1)
        # Velocity in target orbit
        v2 = math.sqrt(mu / r2)
        
        # Transfer orbit semi-major axis
        a_transfer = (r1 + r2) / 2
        
        # Velocity at periapsis of transfer orbit
        v_transfer_peri = math.sqrt(mu * ((2/r1) - (1/a_transfer)))
        # Velocity at apoapsis of transfer orbit
        v_transfer_apo = math.sqrt(mu * ((2/r2) - (1/a_transfer)))
        
        delta_v1 = abs(v_transfer_peri - v1)
        delta_v2 = abs(v2 - v_transfer_apo)
        total_delta_v = delta_v1 + delta_v2
        
        return {
            "initial_orbit_velocity_ms": v1,
            "target_orbit_velocity_ms": v2,
            "delta_v1_ms": delta_v1,
            "delta_v2_ms": delta_v2,
            "total_delta_v_ms": total_delta_v
        }

    def calculate_time_dilation(self, velocity_v: float, time_t: float):
        """
        Calculates time dilation based on Special Relativity.
        velocity_v: Velocity of the moving object (m/s)
        time_t: Time passed for the stationary observer (seconds)
        """
        c = 299792458 # Speed of light (m/s)
        if velocity_v >= c:
            return "Error: Velocity cannot exceed or equal the speed of light."
            
        lorentz_factor = 1 / math.sqrt(1 - (velocity_v**2 / c**2))
        time_moving = time_t / lorentz_factor
        
        return {
            "lorentz_factor": lorentz_factor,
            "stationary_time_s": time_t,
            "moving_object_time_s": time_moving,
            "time_difference_s": time_t - time_moving
        }
