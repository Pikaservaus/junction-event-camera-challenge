import numpy as np
from sklearn.cluster import DBSCAN


"""Real-time event clustering for event camera data."""

from typing import Optional
import numpy as np
from sklearn.cluster import DBSCAN


class SimpleClusterer:
    """
    Simple synchronous clusterer for event camera data.
    """
    
    def __init__(
        self,
        eps: float = 15.0,
        min_samples: int = 50,
        min_events: int = 200,
        flatten_time: bool = False,
    ):
        """        
        Args:
            eps: DBSCAN epsilon parameter (spatial proximity in pixels)
            min_samples: DBSCAN min_samples parameter
            min_events: Minimum events required to perform clustering
            flatten_time: If True, collapse duplicate (x,y) coordinates to reduce data size
        """
        self.eps = eps
        self.min_samples = min_samples
        self.min_events = min_events
        self.flatten_time = flatten_time
        
        self.total_clustered = 0
        
    def cluster_events(
        self,
        x_coords: np.ndarray,
        y_coords: np.ndarray
    ) -> Optional[list[tuple[float, float]]]:

        # Skip if too few events
        if len(x_coords) < self.min_events:
            return None
            
        
        # Create points array for clustering
        points = np.column_stack((x_coords, y_coords))
        
        if self.flatten_time:
            import time
            t_flatten_start = time.perf_counter()
            points_unique, inverse_indices = np.unique(points, axis=0, return_inverse=True)
            t_flatten_elapsed = (time.perf_counter() - t_flatten_start) * 1000
            points_to_cluster = points_unique
        else:
            points_to_cluster = points
        
        # Run DBSCAN
        try:
            import time
            t_start = time.perf_counter()
            labels = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(points_to_cluster).labels_
            t_elapsed = (time.perf_counter() - t_start) * 1000
        except Exception as e:
            print(f"[Clustering] DBSCAN ERROR: {e}")
            return None
            
        # Extract centroids from clusters
        centroids = []
        allClusterPoints = []
        for label in set(labels):
            if label == -1:
                continue  # Skip noise
            cluster_points = points_to_cluster[labels == label]
            centroid = cluster_points.mean(axis=0)
            centroids.append((float(centroid[0]), float(centroid[1])))
            allClusterPoints.append(cluster_points)
        
        
        self.total_clustered += 1
        #return centroids of clusters and all cluster points
        return centroids if centroids else (None, None), allClusterPoints