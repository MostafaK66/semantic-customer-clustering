from utils import DataPreprocessor
from anomaly_detector import AnomalyDetector
from k_means_clustering import KMeansClustering
import settings
import plotting


def main():
    preprocessor = DataPreprocessor(settings.file_path)
    detector = AnomalyDetector()
    plotter = plotting.DataPlotter()
    kmeans_clustering = KMeansClustering()
    df = preprocessor.read_data()
    data = preprocessor.fit_transform(df=df)
    outliers = detector.fit_predict(data)
    data = detector.add_outlier_column(data=data, outliers=outliers)
    data_no_outliers, data_with_outliers = detector.separate_data(data=data)
    plotter.plot_ecdf(data, ['num__age', 'num__balance'])
    kmeans_clustering.find_optimal_clusters(data_no_outliers)
    kmeans_clustering.perform_silhouette_analysis(data_no_outliers, range(2, 10))


if __name__ == "__main__":
    main()


