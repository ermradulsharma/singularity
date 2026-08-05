def load_balancer_round_robin(servers: list, requests: list) -> dict:
    """Simulates a Cloud Load Balancer (Round Robin)"""
    distribution = {s: [] for s in servers}
    for i, req in enumerate(requests):
        distribution[servers[i % len(servers)]].append(req)
    return distribution

def calculate_auto_scaling(cpu_usage: float, threshold: float = 80.0, current_instances: int = 2) -> int:
    """Simulates Cloud Auto-scaling logic"""
    if cpu_usage > threshold:
        return current_instances + 1
    elif cpu_usage < 20.0 and current_instances > 1:
        return current_instances - 1
    return current_instances
