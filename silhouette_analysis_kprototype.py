import pandas as pd
import numpy as np
import gower
from sklearn.metrics import silhouette_samples
from kmodes.kprototypes import KPrototypes
import settings
from tqdm import tqdm


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
        range_ = range(2, 5)
        best_silhouette_score = -1
        best_n_clusters = None

        for cluster in tqdm(range_, desc="Running KPrototype Clustering"):
            kprototype = KPrototypes(n_jobs=-1, n_clusters=cluster, init='Huang', random_state=self.random_state)
            cluster_labels = kprototype.fit_predict(sampled_df, categorical=categorical_columns_index)
            cost.append(kprototype.cost_)

            if cluster > 1:
                distance_matrix = self.calculate_distance_matrix(sampled_df)
                silhouette_vals = silhouette_samples(distance_matrix, cluster_labels, metric='precomputed')
                avg_silhouette = np.mean(silhouette_vals)

                if avg_silhouette > best_silhouette_score:
                    best_silhouette_score = avg_silhouette
                    best_n_clusters = cluster

                print(f"Cluster: {cluster}, Silhouette Score: {avg_silhouette}")

        print(f"Best silhouette score: {best_silhouette_score} with n_clusters: {best_n_clusters}")
        return pd.DataFrame({'Cluster': range_, 'Cost': cost})





