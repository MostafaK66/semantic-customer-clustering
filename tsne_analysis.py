from sklearn.manifold import TSNE
import settings
import pandas as pd

class TSNEAnalysis:
    def __init__(self, random_state=settings.RANDOM_STATE):
        print('started t-SNE')
        self.random_state = random_state

    def get_tsne_3d(self, df, predict):
        sampling_data = df.sample(frac=0.1, replace=True, random_state=self.random_state)
        sampling_clusters = pd.DataFrame(predict).sample(frac=0.1, replace=True, random_state=self.random_state)[0].values
        tsne_3d_object = TSNE(
            n_components=settings.N_COMPONENTS,
            learning_rate=settings.TSNE_LEARNING_RATE,
            init='random',
            perplexity=settings.TSNE_PERPLEXITY,
            n_iter=settings.TSNE_N_ITER,
            random_state=self.random_state
        )

        tsne_results = tsne_3d_object.fit_transform(sampling_data)
        df_tsne_3d = pd.DataFrame(tsne_results, columns=["comp1", "comp2", "comp3"])
        df_tsne_3d["cluster"] = sampling_clusters
        print('end t-SNE')

        return tsne_3d_object, df_tsne_3d



