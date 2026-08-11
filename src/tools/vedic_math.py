def nikhilam_multiply(a: int, b: int) -> int:
    """Fast multiplication using Vedic Nikhilam sutra for numbers near a base. Optimizes large number multiplication deterministically."""
    # Find the nearest power of 10 base
    max_len = len(str(max(a, b)))
    base = 10 ** max_len
    
    # Calculate deficits
    dev_a = base - a
    dev_b = base - b
    
    # Cross subtraction and vertical multiplication
    left = a - dev_b
    right = dev_a * dev_b
    
    return left * base + right
