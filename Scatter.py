import matplotlib.pyplot as plt
import numpy as np

def showOnScatter(x, y):
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='blue', marker='o', edgecolors='k')
    plt.title('Scatter Plot of 2D Vectors')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.grid(True)
    plt.show()