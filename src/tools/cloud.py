class CloudEngine:
    def __init__(self, initial_servers: list):
        self.servers = initial_servers
        self.round_robin_index = 0
        self.active_load = {s: 0.0 for s in self.servers}

    def route_request_round_robin(self, request_payload: str, request_cost: float = 1.0) -> dict:
        """Stateful Load Balancer: Routes a request using strict Round Robin and updates active load."""
        if not self.servers:
            return {"error": "No healthy servers available."}
            
        target_server = self.servers[self.round_robin_index]
        self.active_load[target_server] += request_cost
        
        # Advance index circularly
        self.round_robin_index = (self.round_robin_index + 1) % len(self.servers)
        
        return {
            "routed_to": target_server,
            "request": request_payload,
            "current_cluster_load": dict(self.active_load)
        }

    def calculate_auto_scaling(self, max_threshold: float = 80.0, min_threshold: float = 20.0) -> dict:
        """Dynamically scales server array based on real-time tracked active_load."""
        if not self.servers:
            return {"action": "scale_up", "reason": "Cluster empty", "new_count": 1}
            
        avg_load = sum(self.active_load.values()) / len(self.servers)
        
        if avg_load > max_threshold:
            new_server = f"Server-{len(self.servers) + 1}"
            self.servers.append(new_server)
            self.active_load[new_server] = 0.0
            return {"action": "scale_up", "reason": f"Avg Load ({avg_load:.1f}) > {max_threshold}", "cluster_size": len(self.servers)}
            
        elif avg_load < min_threshold and len(self.servers) > 1:
            removed_server = self.servers.pop()
            del self.active_load[removed_server]
            # Ensure round robin index stays in bounds
            self.round_robin_index = min(self.round_robin_index, len(self.servers) - 1)
            return {"action": "scale_down", "reason": f"Avg Load ({avg_load:.1f}) < {min_threshold}", "cluster_size": len(self.servers)}
            
        return {"action": "none", "reason": "Load stable", "cluster_size": len(self.servers)}
