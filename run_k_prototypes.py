from utils import DataPreprocessor
import settings
from anomaly_detector import AnomalyDetector
from kprototype_clustering import KPrototypeClustering
from plotting import DataPlotter
from silhouette_analysis_kprototype import SilhouetteAnalysisKPrototype
from mca_analysis import MCAAnalysis
import time


def main():
    start_time = time.time()
    preprocessor = DataPreprocessor(settings.file_path)
    detector = AnomalyDetector()
    prototype_clustering = KPrototypeClustering()
    plotter = DataPlotter()
    silhouette_analysis = SilhouetteAnalysisKPrototype()
    mca_analysis = MCAAnalysis()
    df = preprocessor.read_data()
    data = preprocessor.fit_transform(df=df)
    outliers = detector.fit_predict(data)
    df = detector.add_outlier_column(data=df, outliers=outliers)
    df_no_outliers, df_with_outliers, indices_no_outliers, indices_with_outliers = detector.separate_data(data=df)
    transformed_columns = preprocessor.transform_columns(df_no_outliers, ["age", "balance"])
    df_no_outliers_norm = preprocessor.integrate_transformed_columns(df_no_outliers, transformed_columns,
                                                                     ["age", "balance"])
    categorical_columns, categorical_columns_index = preprocessor.identify_and_index_categorical_columns(df_no_outliers_norm)
    sampled_df = silhouette_analysis.sample_dataframe(df=df_no_outliers_norm, frac=settings.SAMPLING_FRACTION)
    silhouette_scores_mixed = silhouette_analysis.find_optimal_clusters(sampled_df=sampled_df, categorical_columns_index=categorical_columns_index)
    prototype_clustering.determine_optimal_clusters(silhouette_scores_mixed=silhouette_scores_mixed)
    clusters_kprototype = prototype_clustering.fit_predict_kprototypes(df=sampled_df, categorical_columns_index=categorical_columns_index)
    mca, mca_3d_df = mca_analysis.get_MCA_3d(df=sampled_df, predict=clusters_kprototype)
    plotter.plot_3d(df=mca_3d_df, title="MCA Space")

    end_time = time.time()
    total_time_minutes = (end_time - start_time) / 60
    print(f"Total running time for KPrototype Clustering: {total_time_minutes:.2f} minutes")

    print('yes')


if __name__ == "__main__":
    main()