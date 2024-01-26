from utils import DataPreprocessor
from anomaly_detector import AnomalyDetector
from silhouette_analysis import SilhouetteAnalysis
import settings
import plotting
import warnings
warnings.filterwarnings("ignore", message="The default value of `n_init` will change from 10 to 'auto' in 1.4. Set "
                                          "the value of `n_init` explicitly to suppress the warning")


def main():
    preprocessor = DataPreprocessor(settings.file_path)
    detector = AnomalyDetector()
    plotter = plotting.DataPlotter()
    silhoutte_analysis = SilhouetteAnalysis()
    df = preprocessor.read_data()
    data = preprocessor.fit_transform(df=df)
    outliers = detector.fit_predict(data)
    data = detector.add_outlier_column(data=data, outliers=outliers)
    data_no_outliers, data_with_outliers = detector.separate_data(data=data)
    plotter.plot_ecdf(data, ['num__age', 'num__balance'])
    silhoutte_analysis.find_optimal_clusters(data=data_no_outliers, k_range=settings.K_RANGE)
    results_df = silhoutte_analysis.perform_combined_silhouette_analysis(data_no_outliers, k_range=settings.K2_RANGE)

    print('yes')


if __name__ == "__main__":
    main()


