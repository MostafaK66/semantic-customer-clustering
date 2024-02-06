import pandas as pd
from kmodes.kprototypes import KPrototypes
import settings
from tqdm import tqdm


class KPrototypeClustering:
    def __init__(self, random_state=settings.RANDOM_STATE):
        self.random_state = random_state

    def find_optimal_clusters(self, df_no_outliers, categorical_columns_index):
        cost = []
        range_ = range(2, 5)

        df_sampled = df_no_outliers.sample(frac=0.1, replace=True, random_state=self.random_state)

        for cluster in tqdm(range_, desc="Running KPrototype Clustering"):
            kprototype = KPrototypes(n_jobs=-1, n_clusters=cluster, init='Huang', random_state=self.random_state)
            kprototype.fit_predict(df_sampled, categorical=categorical_columns_index)
            cost.append(kprototype.cost_)

        return pd.DataFrame({'Cluster': range_, 'Cost': cost})






