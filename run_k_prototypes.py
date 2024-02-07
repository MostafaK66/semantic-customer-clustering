from utils import DataPreprocessor
import settings
from anomaly_detector import AnomalyDetector
from kprototype_clustering import KPrototypeClustering
from plotting import DataPlotter
from silhouette_analysis_kprototype import SilhouetteAnalysisKPrototype

def main():
    preprocessor = DataPreprocessor(settings.file_path)
    detector = AnomalyDetector()
    prototype_clustering = KPrototypeClustering()
    plotter = DataPlotter()
    Silhouette_analysis = SilhouetteAnalysisKPrototype()
    df = preprocessor.read_data()
    data = preprocessor.fit_transform(df=df)
    outliers = detector.fit_predict(data)
    df = detector.add_outlier_column(data=df, outliers=outliers)
    df_no_outliers, df_with_outliers, indices_no_outliers, indices_with_outliers = detector.separate_data(data=df)
    transformed_columns = preprocessor.transform_columns(df_no_outliers, ["age", "balance"])
    df_no_outliers_norm = preprocessor.integrate_transformed_columns(df_no_outliers, transformed_columns,
                                                                     ["age", "balance"])
    categorical_columns, categorical_columns_index = preprocessor.identify_and_index_categorical_columns(df_no_outliers_norm)
    # df_cost = prototype_clustering.find_optimal_clusters(df_no_outliers=df_no_outliers_norm, categorical_columns_index=categorical_columns_index)
    # plotter.plot_clustering_cost(df_cost=df_cost)
    sampled_df = Silhouette_analysis.sample_dataframe(df=df_no_outliers_norm, frac=settings.SAMPLING_FRACTION)
    Silhouette_scores_mixed = Silhouette_analysis.find_optimal_clusters(sampled_df=sampled_df, categorical_columns_index=categorical_columns_index)


    print('yes')



if __name__ == "__main__":
    main()