import prince
import settings
import os
import pandas as pd


class PCAAnalysis:
    def __init__(self, random_state=settings.RANDOM_STATE):
        self.random_state = random_state

    def get_pca_3d(self, df, predict):
        pca_3d_object = prince.PCA(
            n_components=settings.N_COMPONENTS,
            n_iter=settings.PCA_N_ITER,
            rescale_with_mean=True,
            rescale_with_std=True,
            copy=True,
            check_input=True,
            engine='sklearn',
            random_state=self.random_state
        )

        pca_3d_object.fit(df)

        df_pca_3d = pca_3d_object.transform(df)
        df_pca_3d.columns = ["comp1", "comp2", "comp3"]
        df_pca_3d["cluster"] = predict

        return pca_3d_object, df_pca_3d

    def save_eigenvalues_summary(self, pca_3d_object, file_name):

        output_dir = 'output'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        eigenvalues_summary = pca_3d_object.eigenvalues_

        df_eigenvalues = pd.DataFrame(eigenvalues_summary, columns=['eigenvalue'])
        df_eigenvalues['component'] = df_eigenvalues.index + 1

        csv_file_path = os.path.join(output_dir, file_name)
        df_eigenvalues.to_csv(csv_file_path, index=False)

        print(f"Eigenvalues summary saved to {csv_file_path}")

