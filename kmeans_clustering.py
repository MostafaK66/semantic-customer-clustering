from sklearn.cluster import KMeans

class KMeansClustering:
    def __init__(self, init='k-means++', n_init=10, max_iter=100, random_state=42):
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.km = None

    def determine_optimal_clusters(self, silhouette_results):
        max_silhouette_idx = silhouette_results['silhouette_avg'].idxmax()
        self.n_clusters = int(silhouette_results.loc[max_silhouette_idx, 'n_clusters'])
        print(f"Optimal number of clusters determined to be: {self.n_clusters}")

    def fit_predict(self, data):
        # if self.km is None:
        #     raise ValueError("Optimal number of clusters not set. Call determine_optimal_clusters first.")

        self.km = KMeans(n_clusters=self.n_clusters,
                         init=self.init,
                         n_init=self.n_init,
                         max_iter=self.max_iter,
                         random_state=self.random_state)
        return self.km.fit_predict(data)
