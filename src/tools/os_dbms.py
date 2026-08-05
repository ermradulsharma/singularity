def srtf_scheduling(processes: list) -> list:
    """Shortest Remaining Time First (SRTF) OS scheduling algorithm"""
    return sorted(processes, key=lambda x: x['burst_time'])

def optimize_sql_query(query: str) -> str:
    """Simulates DBMS SQL query optimization heuristics"""
    if "SELECT" in query.upper() and "JOIN" in query.upper():
        return "OPTIMIZED: Executed projections before joins to reduce relation size."
    return "QUERY_OK"
