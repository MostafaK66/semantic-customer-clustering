from sklearn.manifold import TSNE
import settings
import pandas as pd
import os

class TSNEAnalysis:
    def __init__(self, random_state=settings.RANDOM_STATE):
        self.random_state = random_state

    def get_tsne_3d(self, df, predict):
        #TODO: Due to computation power I reduced the number of datapoints!!
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

        return tsne_3d_object, df_tsne_3d

    def save_tsne_embeddings(self, tsne_3d_object, df):
        output_dir = 'output'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        tsne_embeddings = tsne_3d_object.fit_transform(df)

        df_tsne = pd.DataFrame(tsne_embeddings, columns=['comp1', 'comp2', 'comp3'])

        csv_file_path = os.path.join(output_dir, 'tsne_embeddings.csv')
        df_tsne.to_csv(csv_file_path, index=False)

        print(f"t-SNE embeddings saved to {csv_file_path}")





