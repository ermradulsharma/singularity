import math
import cmath

class Quantum_Physicist:
    """Agent specialized in Quantum Mechanics, Qubit simulations, and Wave-function analysis."""
    
    def simulate_qubit_superposition(self, alpha_squared: float):
        """
        Simulates the state of a single qubit in superposition.
        alpha_squared is the probability of measuring |0>.
        """
        if not (0 <= alpha_squared <= 1):
            return "Error: Probability must be between 0 and 1."
            
        beta_squared = 1.0 - alpha_squared
        alpha = math.sqrt(alpha_squared)
        beta = math.sqrt(beta_squared)
        
        return {
            "state_vector": f"{alpha:.4f}|0⟩ + {beta:.4f}|1⟩",
            "prob_0": f"{alpha_squared * 100:.2f}%",
            "prob_1": f"{beta_squared * 100:.2f}%",
            "observation": "Quantum state is in superposition until measured."
        }

    def solve_schrodinger_1d_box(self, n: int, L: float):
        """
        Calculates the energy of a particle in a 1D box (Infinite potential well).
        n: Principal quantum number (n >= 1)
        L: Length of the box (in meters)
        """
        if n < 1 or L <= 0:
            return "Error: Invalid quantum state or box length."
            
        # Constants
        h = 6.626e-34 # Planck's constant (J*s)
        m = 9.109e-31 # Mass of electron (kg)
        
        # Energy formula: E_n = (n^2 * h^2) / (8 * m * L^2)
        energy_joules = (n**2 * h**2) / (8 * m * L**2)
        energy_ev = energy_joules / 1.602e-19
        
        return {
            "quantum_state": n,
            "box_length_m": L,
            "energy_Joules": f"{energy_joules:.4e} J",
            "energy_eV": f"{energy_ev:.4f} eV"
        }
