import time
import numpy as np

def compute_metrics(path, start_time, collision=False, goal=None):
    end_time = time.time()
    duration = (end_time - start_time) * 1000

    success = path is not None and len(path) > 0
    path_length = len(path) if path else 0

    collision_rate = 1.0 if collision else 0.0

    if success and goal is not None:
        start = np.array(path[0])
        end = np.array(path[-1])

        straight_dist = np.linalg.norm(end - start)

        path_dist = 0
        for i in range(1, len(path)):
            path_dist += np.linalg.norm(np.array(path[i]) - np.array(path[i-1]))

        path_efficiency = straight_dist / (path_dist + 1e-8)
    else:
        path_efficiency = 0.0

    return {
        "success": success,
        "success_rate": 1.0 if success else 0.0,
        "collision_rate": collision_rate,
        "path_length": path_length,
        "path_efficiency": float(path_efficiency),
        "arrival_time_ms": duration if success else None,
        "compute_time_ms": duration
    }