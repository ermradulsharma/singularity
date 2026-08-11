# Basic Periodic Table weights (g/mol)
ATOMIC_WEIGHTS = {
    'H': 1.008, 'He': 4.0026, 'Li': 6.94, 'Be': 9.0122, 'B': 10.81, 'C': 12.011,
    'N': 14.007, 'O': 15.999, 'F': 18.998, 'Ne': 20.180, 'Na': 22.990, 'Mg': 24.305,
    'Al': 26.982, 'Si': 28.085, 'P': 30.974, 'S': 32.06, 'Cl': 35.45, 'K': 39.098,
    'Ar': 39.95, 'Ca': 40.078, 'Fe': 55.845, 'Cu': 63.546, 'Zn': 65.38, 'Br': 79.904,
    'Ag': 107.87, 'I': 126.90, 'Au': 196.97, 'Hg': 200.59, 'Pb': 207.2
}

def transcribe_dna_to_rna(dna_sequence: str) -> str:
    """Transcribes a DNA sequence into an RNA sequence (T -> U)."""
    return dna_sequence.upper().replace('T', 'U')

def reverse_complement_dna(dna_sequence: str) -> str:
    """Generates the reverse complement of a DNA sequence."""
    complement_map = str.maketrans('ATCGatcg', 'TAGCtagc')
    return dna_sequence.translate(complement_map)[::-1]

def calculate_molecular_weight(formula: str) -> float:
    """Calculates the molecular weight of a simple chemical formula. Supports symbols and counts, e.g., 'H2O', 'CO2', 'C6H12O6'. (Does not support nested parentheses in this basic version)."""
    import re
    weight = 0.0
    # Matches an Element symbol followed optionally by a number
    pattern = re.compile(r'([A-Z][a-z]?)([0-9]*)')
    matches = pattern.findall(formula)
    
    for element, count in matches:
        if element not in ATOMIC_WEIGHTS:
            raise ValueError(f"Unknown element: {element}")
        
        count = int(count) if count else 1
        weight += ATOMIC_WEIGHTS[element] * count
        
    return weight
