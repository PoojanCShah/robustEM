import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.datasets import make_blobs, load_iris
from scipy.stats import multivariate_normal
from main import robustEM
from sklearn.decomposition import PCA


# Generate synthetic Gaussian data

n_samples = 500
n_features = 2
true_clusters = [(0,0),(5,0),(10,10),(15,0),(20,0)]
X, y_true = make_blobs(n_samples=n_samples, centers=true_clusters, cluster_std=0.60, random_state=42)


# Run robust EM

alpha, mu, sigma, z = robustEM(X)


# Predict cluster assignment

predicted_labels = np.argmax(z, axis=0)

# Function to plot ellipse for a Gaussian

def ellipse(ax, mean, cov, std_multiplier=2.0, **kwargs):
    """Draws an ellipse representing the Gaussian distribution."""
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * std_multiplier * np.sqrt(vals)
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=theta, **kwargs)
    ax.add_patch(ellipse)


# Plot data, cluster means, and Gaussian boundaries

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(X[:, 0], X[:, 1], c=predicted_labels, s=40, cmap='viridis', alpha=0.6, label="Data points")
ax.scatter(mu[:, 0], mu[:, 1], c='red', s=100, marker='x', label='Estimated Means')

for i in range(len(mu)):
    ellipse(ax, mu[i], sigma[i], std_multiplier=2.0, edgecolor='red', facecolor='none', linewidth=2)
    ax.text(mu[i, 0], mu[i, 1], f'{i}', fontsize=9, color='red', ha='right', va='bottom')

ax.set_title("Robust EM Clustering")
ax.legend()
ax.grid(True)
plt.axis('equal')
plt.savefig("plot.png")
