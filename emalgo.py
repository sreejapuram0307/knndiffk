import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

X, y = make_blobs(
    n_samples=300, centers=4, random_state=0
)

kmeans = KMeans(n_clusters=4, random_state=0, n_init=10)
k_labels = kmeans.fit_predict(X)

gmm = GaussianMixture(n_components=4, random_state=0)
gmm.fit(X)
g_labels = gmm.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=k_labels)
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            marker="X", s=200, edgecolors="black")
plt.title("K-Means Clustering")
plt.show()

plt.scatter(X[:, 0], X[:, 1], c=g_labels)
plt.title("Gaussian Mixture Model Clustering")
plt.show()

print("K-Means Silhouette Score:", silhouette_score(X, k_labels))
print("GMM Silhouette Score:", silhouette_score(X, g_labels))

probabilities = gmm.predict_proba(X[:3])
print("GMM probabilities for first 3 samples:\n", probabilities)