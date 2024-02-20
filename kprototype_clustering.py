from kmodes.kprototypes import KPrototypes
import settings


class KPrototypeClustering:
    def __init__(self, n_init=settings.N_INIT, max_iter=settings.MAX_ITER, gamma=settings.GAMMA, init=settings.KPROTOTYPE_INIT, random_state=settings.RANDOM_STATE):
        self.random_state = random_state
        self.n_init = n_init
        self.max_iter = max_iter
        self.gamma = gamma
        self.init = init
        self.n_clusters = None
        self.kprototype = None

    def determine_optimal_clusters(self, silhouette_scores_mixed):
        max_silhouette_idx = silhouette_scores_mixed['silhouette_avg'].idxmax()
        self.n_clusters = int(silhouette_scores_mixed.loc[max_silhouette_idx, 'n_clusters'])
        print(f"Optimal number of clusters for KPrototype Clustering: {self.n_clusters}")

    def fit_predict_kprototypes(self, df, categorical_columns_index):
        if self.n_clusters is None:
            raise ValueError("Number of clusters not set. Call 'determine_optimal_clusters' first.")

        self.kprototype = KPrototypes(n_jobs=-1, n_init=self.n_init, n_clusters=self.n_clusters, gamma=self.gamma, max_iter=self.max_iter, init=self.init,
                                      random_state=self.random_state)
        self.kprototype.fit(df, categorical=categorical_columns_index)

        return self.kprototype.predict(df, categorical=categorical_columns_index)














