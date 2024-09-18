from sklearn.cluster import KMeans
import numpy as np

# Example data
X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])

# K-means clustering
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)

# Predicting the cluster for each data point
labels = kmeans.labels_

# Cluster centers
centers = kmeans.cluster_centers_

print("Labels:", labels)
print("Centers:", centers)
