import numpy as np
from numba import jit


@jit(nopython=True)
def _inertia(classifications,positions,num_clusters):
    centroids_positions = np.zeros(num_clusters,len(positions[0]))
    centroids_num = np.zeros(num_clusters)
    for i in range(len(classifications)):
        class_ = classifications[i]
        centroids_positions[class_] += positions[i]
        centroids_num[class_] += 1
    centroids_mean = centroids_positions/centroids_num
    inertia = 0
    for i in range(len(classifications)):
        class_ = classifications[i]
        inertia+=np.sqrt(np.sum((positions[i]-centroids_positions[class_])**2))

    return inertia

class InertiaObjective:
    def __init__(self,positions,num_clusters):
        self.positions = positions
        self.num_clusters = num_clusters

    def compute(self,classifications):
        return _inertia(classifications,self.positions,self.num_clusters)
