import pandas as pd
import numpy as np
import gower
from sklearn.metrics import silhouette_samples
from kmodes.kprototypes import KPrototypes
import settings
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
import os


class SilhouetteAnalysisKPrototype:
    def __init__(self, random_state=settings.RANDOM_STATE):
        self.random_state = random_state

    def sample_dataframe(self, df, frac):
        return df.sample(frac=frac, replace=True, random_state=self.random_state)

    def calculate_distance_matrix(self, df):
        df_array = df.to_numpy()
        return gower.gower_matrix(df_array)

    def find_optimal_clusters(self, sampled_df, categorical_columns_index):
        cost = []
        range_ = settings.K_PROTOTYPE_RANGE

        silhouette_scores = []

        for cluster in tqdm(range_, desc="Running KPrototype Clustering"):
            kprototype = KPrototypes(n_jobs=-1, n_clusters=cluster, init='Huang', random_state=self.random_state)
            cluster_labels = kprototype.fit_predict(sampled_df, categorical=categorical_columns_index)
            cost.append(kprototype.cost_)

            if cluster > 1:
                distance_matrix = self.calculate_distance_matrix(sampled_df)
                silhouette_vals = silhouette_samples(distance_matrix, cluster_labels, metric='precomputed')
                avg_silhouette = np.mean(silhouette_vals)

                silhouette_scores.append((cluster_labels, silhouette_vals, avg_silhouette, cluster))


        self.plot_silhouette(silhouette_scores)
        return pd.DataFrame({'n_clusters': range_, 'silhouette_avg': cost})

    def plot_silhouette(self, silhouette_scores):
        n_rows = len(silhouette_scores)
        fig, axs = plt.subplots(n_rows, 1, figsize=(7, n_rows * 5))

        if n_rows == 1:
            axs = [axs]

        for idx, (cluster_labels, sample_silhouette_values, silhouette_avg, n_clusters) in enumerate(silhouette_scores):
            self._plot_silhouette(axs[idx], cluster_labels, sample_silhouette_values, silhouette_avg, n_clusters)

        plt.tight_layout()
        output_dir = 'output'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plot_path = os.path.join(output_dir, 'Combined_Silhouette_plot_mixed.png')
        plt.savefig(plot_path)
        plt.close()

    def _plot_silhouette(self, ax, cluster_labels, sample_silhouette_values, silhouette_avg, n_clusters):
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







