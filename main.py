import KmeansFunctions as kmeans
import numpy as np
import Scatter as s

# Example dataset
X = [[1, 2], [2, 3], [3, 4], [5, 7], [3, 2], [8, 10], [6, 8], [9, 11], [8, 6], [4, 2]]
# Convert to NumPy array for easier manipulation
X = np.array(X)

# Separate x and y components
x_values = X[:, 0]  # First column, x coordinates
y_values = X[:, 1]  # Second column, y coordinates

k_vlues, inertia_values = kmeans.Kvalue(X)


s.showOnScatter(x_values, y_values)
s.showOnScatter(k_vlues, inertia_values)

print(kmeans.Kmeans(X, 3))