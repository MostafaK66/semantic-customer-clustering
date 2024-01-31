from utils import DataPreprocessor
from anomaly_detector import AnomalyDetector
from silhouette_analysis import SilhouetteAnalysis
import settings
import plotting
import warnings
from kmeans_clustering import KMeansClustering
from pca_analysis import PCAAnalysis
from tsne_analysis import TSNEAnalysis
warnings.simplefilter(action='ignore', category=FutureWarning)


def main():
    preprocessor = DataPreprocessor(settings.file_path)
    detector = AnomalyDetector()
    plotter = plotting.DataPlotter()
    silhoutte_analysis = SilhouetteAnalysis()
    kmeans_clustering = KMeansClustering()
    pca_analysis = PCAAnalysis()
    tsne_analysis = TSNEAnalysis()
    df = preprocessor.read_data()
    data = preprocessor.fit_transform(df=df)
    outliers = detector.fit_predict(data)
    data = detector.add_outlier_column(data=data, outliers=outliers)
    data_no_outliers, data_with_outliers = detector.separate_data(data=data)
    plotter.plot_ecdf(data, ['num__age', 'num__balance'])
    silhoutte_analysis.find_optimal_clusters(data=data_no_outliers, k_range=settings.K_RANGE)
    silhoutte_results = silhoutte_analysis.perform_combined_silhouette_analysis(data_no_outliers, k_range=settings.K2_RANGE)
    kmeans_clustering.determine_optimal_clusters(silhoutte_results)
    clusters_predict = kmeans_clustering.fit_predict(data_no_outliers)
    pca_3d_object, df_pca_3d = pca_analysis.get_pca_3d(df=data_no_outliers, predict=clusters_predict)
    plotter.plot_3d(df=df_pca_3d, title="PCA Space")
    pca_analysis.save_eigenvalues_summary(pca_3d_object)
    tsne_3d_object, df_tsne_3d = tsne_analysis.get_tsne_3d(df=data_no_outliers, predict=clusters_predict)
    plotter.plot_3d(df=df_tsne_3d, title="t-SNE Space")
    tsne_analysis.save_tsne_embeddings(tsne_3d_object=tsne_3d_object, df=df_tsne_3d)

    print('yes')


if __name__ == "__main__":
    main()


