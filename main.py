from utils import DataPreprocessor
from anomaly_detector import AnomalyDetector
import settings


def main():
    preprocessor = DataPreprocessor(settings.file_path)
    detector = AnomalyDetector()
    df = preprocessor.read_data()
    data = preprocessor.fit_transform(df=df)
    outliers = detector.fit_predict(data)
    data = detector.add_outlier_column(data=data, outliers=outliers)
    data_no_outliers, data_with_outliers = detector.separate_data(data=data)
    print(data_no_outliers, data_with_outliers)


if __name__ == "__main__":
    main()


