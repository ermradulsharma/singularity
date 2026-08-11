def srtf_scheduling(processes: list) -> list:
    """Shortest Remaining Time First (SRTF) OS scheduling algorithm"""
    return sorted(processes, key=lambda x: x['burst_time'])

def optimize_sql_query(query: str) -> dict:
    """Parses a SQL query heuristically to estimate execution cost (Big-O) based on simulated operations."""
    query_upper = query.upper()
    
    # Base execution cost for scanning
    cost = 1
    operations = []
    
    if "SELECT" in query_upper:
        operations.append("Projection Scan")
        
    if "JOIN" in query_upper:
        # Nested loop join worst case complexity O(n^2)
        cost *= 100 
        operations.append("Nested Loop Join Detected (O(n^2))")
        
    if "WHERE" in query_upper:
        cost *= 0.5 # Heuristic: Filtering reduces dataset size
        operations.append("Filter Condition Applied (Reduces cardinality)")
        
    if "ORDER BY" in query_upper:
        # Sorting cost O(n log n)
        cost += 50
        operations.append("Sort Operation (O(n log n))")
        
    # Heuristic optimization recommendation
    recommendation = "QUERY_OK"
    if "JOIN" in query_upper and "WHERE" not in query_upper:
        recommendation = "WARNING: Unfiltered JOIN. Consider adding WHERE clauses or indices to prevent Cartesian products."
    elif "SELECT *" in query_upper:
        recommendation = "WARNING: Avoid SELECT *. Explicitly list required columns to reduce I/O overhead."

    return {
        "estimated_relative_cost": cost,
        "operations_detected": operations,
        "recommendation": recommendation
    }
