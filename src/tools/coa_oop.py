def cache_hit_ratio(hits: int, misses: int) -> float:
    """Computer Organization: Calculates Cache Hit Ratio"""
    total = hits + misses
    return hits / total if total > 0 else 0.0

class OOP_Simulator:
    """Object Oriented Programming: Encapsulation logic"""
    def __init__(self, state):
        self._encapsulated_state = state # Private state
        
    def get_state(self):
        return self._encapsulated_state
    
    def set_state(self, new_state):
        self._encapsulated_state = new_state
