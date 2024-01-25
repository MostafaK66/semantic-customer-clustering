import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from yellowbrick.cluster import KElbowVisualizer


class KMeansClustering:
    def __init__(self, random_state=123, n_init='auto'):
        self.km = KMeans(init="k-means++", random_state=random_state, n_init=n_init)

    def find_optimal_clusters(self, data, k_range=(2, 10)):
        self.visualizer = KElbowVisualizer(self.km, k=k_range)
        self.visualizer.fit(data)

        output_dir = 'output'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        plot_path = os.path.join(output_dir, 'elbow_method_plot.png')
        self.visualizer.show(outpath=plot_path)
        print(f"Elbow method plot saved to {plot_path}")

    def silhouette_analysis(self, data, n_clusters):
        clusterer = KMeans(n_clusters=n_clusters, random_state=123)
        cluster_labels = clusterer.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        print(f"For n_clusters = {n_clusters}, the average silhouette_score is: {silhouette_avg}")

        sample_silhouette_values = silhouette_samples(data, cluster_labels)
        self.plot_silhouette(data, n_clusters, cluster_labels, sample_silhouette_values, silhouette_avg)

    def plot_silhouette(self, data, n_clusters, cluster_labels, sample_silhouette_values, silhouette_avg):
        plt.figure(figsize=(10, 7))
        plt.xlim([-0.1, 1])
        plt.ylim([0, len(data) + (n_clusters + 1) * 10])

        y_lower = 10
        for i in range(n_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            color = cm.nipy_spectral(float(i) / n_clusters)
            plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)
            plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10

        plt.title(f"The Silhouette Plot for n_clusters = {n_clusters}", fontsize=14)
        plt.xlabel("The silhouette coefficient values")
        plt.ylabel("Cluster label")
        plt.axvline(x=silhouette_avg, color="red", linestyle="--")
        plt.yticks([])
        plt.xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        output_dir = 'output'
        plot_path = os.path.join(output_dir, f'Silhouette_plot_{n_clusters}.png')
        plt.savefig(plot_path)
        print(f"Silhouette plot for n_clusters = {n_clusters} saved to {plot_path}")
        plt.close()

    def perform_silhouette_analysis(self, data, k_range):
        for n_clusters in k_range:
            self.silhouette_analysis(data, n_clusters)


