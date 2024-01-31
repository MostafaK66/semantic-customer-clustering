import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from yellowbrick.cluster import KElbowVisualizer
from tqdm import tqdm
import settings


class SilhouetteAnalysis:
    def __init__(self, random_state=settings.RANDOM_STATE, n_init='auto'):
        self.km = KMeans(init="k-means++", random_state=random_state, n_init=n_init)

    def find_optimal_clusters(self, data, k_range):
        self.visualizer = KElbowVisualizer(self.km, k=k_range)
        self.visualizer.fit(data)

        output_dir = 'output'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        plot_path = os.path.join(output_dir, 'elbow_method_plot.png')
        self.visualizer.show(outpath=plot_path)
        print(f"Elbow method plot saved to {plot_path}")

    def perform_combined_silhouette_analysis(self, data, k_range):
        results_df = pd.DataFrame(columns=['n_clusters', 'silhouette_avg'])
        n_rows = len(k_range)
        plt.figure(figsize=(10, 7 * n_rows))

        for idx, n_clusters in tqdm(enumerate(k_range), total=len(k_range), desc="Analyzing Silhouette Scores"):
            ax = plt.subplot(n_rows, 1, idx + 1)
            clusterer = KMeans(n_clusters=n_clusters, random_state=123)
            cluster_labels = clusterer.fit_predict(data)
            silhouette_avg = silhouette_score(data, cluster_labels)
            sample_silhouette_values = silhouette_samples(data, cluster_labels)

            new_row = pd.DataFrame({'n_clusters': [n_clusters], 'silhouette_avg': [silhouette_avg]})
            results_df = pd.concat([results_df, new_row], ignore_index=True)

            self.plot_silhouette(ax, cluster_labels, sample_silhouette_values, silhouette_avg, n_clusters)

        plt.tight_layout()
        output_dir = 'output'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plot_path = os.path.join(output_dir, 'Combined_Silhouette_plot.png')
        plt.savefig(plot_path)
        plt.close()

        return results_df

    def plot_silhouette(self, ax, cluster_labels, sample_silhouette_values, silhouette_avg, n_clusters):
        y_lower = 10
        for i in range(n_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            color = cm.nipy_spectral(float(i) / n_clusters)
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10

        ax.set_xlim([-0.1, 1])
        ax.set_ylim([0, len(cluster_labels) + (n_clusters + 1) * 10])
        ax.set_title(f"The Silhouette Plot for n_clusters = {n_clusters}", fontsize=14)
        ax.set_xlabel("The silhouette coefficient values")
        ax.set_ylabel("Cluster label")
        ax.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax.set_yticks([])
        ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])






