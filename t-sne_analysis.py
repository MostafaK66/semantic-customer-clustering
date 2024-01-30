from sklearn.manifold import TSNE


import settings
import os
import pandas as pd


class TSNEAnalysis:
    def __init__(self, random_state=settings.RANDOM_STATE):
        self.random_state = random_state

    def get_tsne_3d(self, df, predict):
        tsne_3d_object = TSNE(
            n_components=settings.N_COMPONENTS,
            learning_rate=settings.TSNE_LEARNING_RATE,
            init='random',
            perplexity=settings.TSNE_PERPLEXITY,
            n_iter=settings.TSNE_N_ITER,
            random_state=self.random_state
        )

        df_tsne_3d = tsne_3d_object.fit_transform(df)
        df_tsne_3d.columns = ["comp1", "comp2", "comp3"]
        df_tsne_3d["cluster"] = predict
        
        return tsne_3d_object, df_tsne_3d