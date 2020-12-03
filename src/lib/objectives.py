import numpy as np
from numba import jit


@jit(nopython=True)
def _inertia(classifications,points,num_clusters,num_features):
    centroids_points = np.zeros((num_clusters,num_features))
    centroids_mean = np.empty((num_clusters,num_features))
    centroids_num = np.zeros(num_clusters)
    for i in range(len(classifications)):
        class_ = classifications[i]
        centroids_points[class_] += points[i]
        centroids_num[class_] += 1

    for i in range(num_clusters):
        if centroids_num[i] != 0:
            centroids_mean[i] = centroids_points[i]/centroids_num[i]
            
    inertia = 0
    for i in range(len(classifications)):
        class_ = classifications[i]
        inertia += np.sqrt(np.sum((points[i]-centroids_mean[class_])**2))

    return inertia

class InertiaObjective:
    def __init__(self,points,num_clusters):
        self.points = points
        self.num_features = len(points[0])
        self.num_clusters = num_clusters

    def compute(self,classifications):
        return _inertia(classifications,self.points,self.num_clusters,self.num_features)
