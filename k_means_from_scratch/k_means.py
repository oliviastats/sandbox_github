import numpy as np
from sklearn.datasets import make_blobs

class kmeans:
    def __init__(self, k=None):
        self.k = k

    def initialize_centroids(self, X, k):
        '''initialize centroids '''
        centroids = []
        dimension = X.shape[1]
        for i in range(k):
            centroid = []
            for d in range(dimension):
                max_value = max(X[d,:])
                min_value = min(X[d,:])
                centroid_dim = np.random.uniform(min_value, max_value)
                centroid.append(centroid_dim)
            centroids.append(np.array(centroid))
        return np.asarray(centroids)

    def euclidean_distance(self, x,y):
        return np.sqrt(np.sum((x-y)**2))

    def assign_cluster(self, X, centroids):
        num_samples = X.shape[0]
        assigned_clusters = []
        for i in range(num_samples):
            distances = []
            for centroid in centroids:
                distance_curr = self.euclidean_distance(centroid, X[i, :])
                distances.append(distance_curr)
            idx_cluster = np.argmin(distances)
            assigned_clusters.append(centroids[idx_cluster])
        return np.asarray(assigned_clusters)
    
    def recompute_centroids(self, X, centroids, assigned_clusters):
        centroids_new = []
        X_same_cluster = np.array([])
        for centroid in centroids:
            cond_same_cluster = (assigned_clusters == centroid)
            X_same_cluster = X[np.logical_and(cond_same_cluster[:,0], cond_same_cluster[:,1])]
            centroids_new.append(np.mean(X_same_cluster, axis=0))
        return np.asarray(centroids_new)

    def run_kmeans(self, k, X):
        centroids_prev = self.initialize_centroids(X, k)
        centroid_dist_change = 100
        while centroid_dist_change >.001:
            assigned_cluster = self.assign_cluster(X, centroids_prev)
            centroids_new = self.recompute_centroids(X, centroids_prev, assigned_cluster)
            centroid_dist_change = self.euclidean_distance(centroids_new, centroids_prev)
            centroids_prev = centroids_new
        return assigned_cluster


if __name__=='__main__':
    kmeans_clustering = kmeans(k=2)
    X_train, _ = make_blobs(n_samples=500, centers=3, n_features=2, random_state=20)
    assigned_cluster = kmeans_clustering.run_kmeans(2, X_train)
