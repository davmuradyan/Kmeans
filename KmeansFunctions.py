import numpy as np
import Scatter as s
# Function to calculate Euclidean distance between two points or arrays
def EuclideanDistance(a, b):
    return np.sqrt(np.sum((np.array(a) - np.array(b)) ** 2))

# Function to calculate mean vector of a cluster
def MeanVector(x, centroid):
    n, k = len(x), len(x[0])

    if n == 1:
        return [(centroid[i] + x[0][i]) / 2 for i in range(k)]

    has_centroid = False
    for i in x:
        if any(np.array_equal(i, element) for element in x):
            has_centroid = True
            break

    if not has_centroid:
        return_val = [(x[1][j] + x[0][j]) / 2 for j in range(k)]
        for i in range(2, n):
            mean = [(return_val[j] + x[i][j]) / 2 for j in range(k)]
            return_val = mean
        return return_val
    else:
        return_val = [(x[1][j] + x[0][j]) / 2 for j in range(k)]
        for i in range(2, n):
            mean = [(return_val[j] + x[i][j]) / 2 for j in range(k)]
            return_val = mean
        mean = [(return_val[j] + centroid[j]) / 2 for j in range(k)]
        return return_val

# Function to initialize centroids using K-means++
def KmeansPP(x, K):
    centroids = [x[0]]
    for i in range(K - 1):
        D2 = np.array([min([EuclideanDistance(a, c) for c in centroids]) for a in x])
        SS = D2.sum()
        probabilities = D2 / SS
        index = np.where(probabilities == max(probabilities))[0][0]
        centroids.append(x[index])
    return centroids

# Function to perform K-means clustering
def Kmeans(x, K):
    centroids = KmeansPP(x, K)
    result = [[] for _ in range(K)]
    for i in range(len(x)):
        distances = [EuclideanDistance(x[i], c) for c in centroids]
        index = np.argmin(distances)
        result[index].append(x[i])
        centroids[index] = MeanVector(result[index], centroids[index])
    return result

# Function to calculate inertia (sum of squared distances within clusters)
def calculate_inertia(clusters):
    inertia = 0.0
    for cluster in clusters:
        centroid = np.mean(cluster, axis=0)
        for point in cluster:
            inertia += np.sum((point - centroid) ** 2)
    return inertia

# Function to evaluate different values of K and their corresponding inertia
def Kvalue(x):
    K_values = []
    inertia_values = []
    for i in range(1, len(x) + 1):
        clusters = Kmeans(x, i)
        inertia = calculate_inertia(clusters)
        K_values.append(i)
        inertia_values.append(inertia)
    return K_values, inertia_values
