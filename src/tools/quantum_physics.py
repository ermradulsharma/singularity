import numpy as np

class Quantum_Physicist:
    """Agent specialized in Quantum Mechanics, Qubit simulations, and Wave-function analysis."""
    
    def apply_hadamard_gate(self, alpha: float, beta: float):
        """Applies a real Hadamard quantum gate matrix to a state vector [alpha, beta]."""
        state = np.array([alpha, beta])
        norm = np.linalg.norm(state)
        if norm == 0:
            return "Error: Invalid state vector."
        state = state / norm
        
        # Real Hadamard Matrix
        H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])
        new_state = np.dot(H, state)
        
        prob_0 = np.abs(new_state[0])**2
        prob_1 = np.abs(new_state[1])**2
        
        return {
            "initial_state": [float(state[0]), float(state[1])],
            "new_state_vector": [float(new_state[0]), float(new_state[1])],
            "prob_0": f"{prob_0 * 100:.2f}%",
            "prob_1": f"{prob_1 * 100:.2f}%"
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
