import heapq

def dijkstra(graph: dict, start: str) -> dict:
    """Dijkstra's Shortest Path Algorithm for Graph Theory"""
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = [(0, start)]
    
    while pq:
        current_distance, current_node = heapq.heappop(pq)
        if current_distance > distances[current_node]:
            continue
            
        for neighbor, weight in graph[current_node].items():
            dist = current_distance + weight
            if dist < distances[neighbor]:
                distances[neighbor] = dist
                heapq.heappush(pq, (dist, neighbor))
                
    return distances

def binary_search(arr: list, target: int) -> int:
    """Binary search implementation"""
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1
