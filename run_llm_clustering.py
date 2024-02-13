from utils import DataPreprocessor
from anomaly_detector import AnomalyDetector
from silhouette_analysis_kmeans import SilhouetteAnalysis
import time
import settings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    start_time = time.time()
    preprocessor = DataPreprocessor(settings.file_path)
    detector = AnomalyDetector()
    df = preprocessor.read_data()
    silhoutte_analysis = SilhouetteAnalysis()
    df_embedding = preprocessor.compile_and_encode_texts(df=df)
    outliers = detector.fit_predict(data=df_embedding)
    data = detector.add_outlier_column(data=df_embedding, outliers=outliers)
    data_no_outliers, data_with_outliers, indices_no_outliers, indices_with_outliers = detector.separate_data(data=data)
    filtered_df = detector.filter_outliers(df=df, index_with_outliers=indices_with_outliers)
    silhoutte_analysis.find_optimal_clusters(data=data_no_outliers, k_range=settings.K_RANGE,
                                             file_name='elbow_llm-kmeans.png')

    end_time = time.time()
    total_time_minutes = (end_time - start_time) / 60
    print(f"Total running time for LLM-Kmeans Clustering: {total_time_minutes:.2f} minutes")


if __name__ == "__main__":
    main()