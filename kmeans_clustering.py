from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Create sample data
X, _ = make_blobs(n_samples=1000, centers=3, n_features=2)
kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(X)

print("Cluster centroids:\n", kmeans.cluster_centers_)

# New point
pred, _ = make_blobs(n_samples=1, centers=1, n_features=2)
print("New data point:\n", pred)
print("New cluster belongs to:", kmeans.predict(pred))

# Plotting
plt.figure(figsize=(8, 6))
colors = ['cyan', 'magenta', 'yellow']

for i in range(3):
    cluster_points = X[kmeans.labels_ == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i], label=f'Cluster {i+1}', s=50)

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', s=200, label='Centroids')
plt.scatter(pred[:, 0], pred[:, 1], c='black', marker='*', s=200, label='New Data Point')

plt.title('KMEANS CLUSTERING')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')  # Fixed label spelling
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
