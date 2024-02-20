from prince import MCA

import settings


class MCAAnalysis:
    def __init__(self, n_components=settings.N_COMPONENTS, n_iter=settings.MCA_N_ITER, random_state=settings.RANDOM_STATE):
        self.n_components = n_components
        self.n_iter = n_iter
        self.random_state = random_state
        self.mca = MCA(n_components=self.n_components, n_iter=self.n_iter, random_state=self.random_state)

    def get_MCA_3d(self, df, predict):
        mca_3d_df = self.mca.fit_transform(df)
        mca_3d_df.columns = ["comp1", "comp2", "comp3"]
        mca_3d_df["cluster"] = predict
        return self.mca, mca_3d_df
