from utils import DataPreprocessor
from anomaly_detector import AnomalyDetector
from silhouette_analysis_kmeans import SilhouetteAnalysis
from kmeans_clustering import KMeansClustering
from pca_analysis import PCAAnalysis
import plotting
import time
import settings
import warnings
from tsne_analysis import TSNEAnalysis
warnings.simplefilter(action='ignore', category=FutureWarning)

def llm_kmeans():
    start_time = time.time()
    preprocessor = DataPreprocessor(settings.file_path)
    detector = AnomalyDetector()
    df = preprocessor.read_data()
    silhoutte_analysis = SilhouetteAnalysis()
    kmeans_clustering = KMeansClustering()
    pca_analysis = PCAAnalysis()
    plotter = plotting.DataPlotter()
    tsne_analysis = TSNEAnalysis()
    df_embedding = preprocessor.compile_and_encode_texts(df=df)
    outliers = detector.fit_predict(data=df_embedding)
    data = detector.add_outlier_column(data=df_embedding, outliers=outliers)
    data_no_outliers, data_with_outliers, indices_no_outliers, indices_with_outliers = detector.separate_data(data=data)
    filtered_df = detector.filter_outliers(df=df, index_with_outliers=indices_with_outliers)
    silhoutte_analysis.find_optimal_clusters(data=data_no_outliers, k_range=settings.K_RANGE,
                                             file_name='elbow_llm-kmeans.png')
    silhoutte_results = silhoutte_analysis.perform_combined_silhouette_analysis(data_no_outliers,
                                                                                k_range=settings.K2_RANGE, file_name='LLM-Kmeans_Silhouette_plot.png')
    kmeans_clustering.determine_optimal_clusters(silhoutte_results)
    clusters_predict = kmeans_clustering.fit_predict(data_no_outliers)
    pca_3d_object, df_pca_3d = pca_analysis.get_pca_3d(df=data_no_outliers, predict=clusters_predict)
    plotter.plot_3d(df=df_pca_3d, title="PCA for LLM-KMeans Space")
    pca_analysis.save_eigenvalues_summary(pca_3d_object, file_name='eigenvalues_LLM-kmeans_pca.csv')
    pca_analysis.save_eigenvalues_summary(pca_3d_object=pca_3d_object, file_name='eigenvalues_llm_kmeans_pca.csv')
    tsne_3d_object, df_tsne_3d = tsne_analysis.get_tsne_3d(df=data_no_outliers, predict=clusters_predict)
    plotter.plot_3d(df=df_tsne_3d, title="t-SNE LLM-Kmeans Space")
    tsne_analysis.save_tsne_embeddings(tsne_3d_object=tsne_3d_object, df=df_tsne_3d,
                                       file_name='tsne_embeddings_llm-kmeans.csv')

    end_time = time.time()
    total_time_minutes = (end_time - start_time) / 60
    print(f"Total running time for LLM-Kmeans Clustering: {total_time_minutes:.2f} minutes")


if __name__ == "__main__":
    llm_kmeans()