import numpy as np
from numba import jit
import numba

@jit(nopython=True)
def _inertia(classifications,points,centroids):
    inertia = 0
    for i in range(len(classifications)):
        class_ = classifications[i]
        inertia += np.sqrt(np.sum((points[i]-centroids[class_])**2))
    return inertia

@jit(nopython=True)
def _compute_centroids_and_inertia(classifications,points,num_clusters,num_features):
    centroids_cumulated = np.zeros((num_clusters,num_features))
    centroids = np.empty((num_clusters,num_features))
    centroids_num = np.zeros(num_clusters)
    for i in range(len(classifications)):
        class_ = classifications[i]
        centroids_cumulated[class_] += points[i]
        centroids_num[class_] += 1

    for i in range(num_clusters):
        if centroids_num[i] != 0:
            centroids[i] = centroids_cumulated[i]/centroids_num[i]

    return _dist(classifications,points,centroids)


@jit(nopython=True)
def compute_classifications(centroids,points):
    classifications = np.empty(len(points),dtype=numba.int32)
    lowest_dist = np.inf
    for i in range(len(points)):
        lowest_dist = np.inf
        for j in range(len(centroids)):
            dist = np.sqrt(np.sum((points[i]-centroids[j])**2))
            if dist < lowest_dist:
                lowest_dist = dist
                classifications[i] = j
    return classifications

# @jit(nopython=True)
# def _compute_classifications_and_inertia(centroids,points,num_clusters,num_features):
#     classifications = np.empty(len(points),dtype=numba.int32)
#     lowest_dist = np.inf
#     for i in range(len(points)):
#         lowest_dist = np.inf
#         for j in range(len(centroids)):
#             dist = np.sqrt(np.sum((points[i]-centroids[j])**2))
#             if dist < lowest_dist:
#                 lowest_dist = dist
#                 classifications[i] = j
#     return _inertia(classifications,points,centroids)

class InertiaObjective:
    def __init__(self,points,num_clusters):
        self.points = points
        self.num_features = len(points[0])
        self.num_clusters = num_clusters

    def compute(self,classifications=None,centroids=None):
        if not isinstance(classifications,type(None)):
            return _compute_centroids_and_inertia(classifications,self.points,self.num_clusters,self.num_features)
        elif not isinstance(centroids,type(None)):
            classifications = compute_classifications(centroids,self.points)
            return _inertia(classifications,self.points,centroids)
        else:
            raise SystemError
