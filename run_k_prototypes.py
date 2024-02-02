from utils import DataPreprocessor
import settings
from anomaly_detector import AnomalyDetector


def main():
    preprocessor = DataPreprocessor(settings.file_path)
    detector = AnomalyDetector()
    df = preprocessor.read_data()
    data_norm = preprocessor.encode_and_transform(df=df)
    outliers = detector.fit_predict(data_norm)
    data = detector.add_outlier_column(data=data_norm, outliers=outliers)
    data_no_outliers, data_with_outliers, indices_no_outliers, indices_with_outliers = detector.separate_data(data=data)

    print('yes')



if __name__ == "__main__":
    main()