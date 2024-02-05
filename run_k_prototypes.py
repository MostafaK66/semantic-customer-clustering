from utils import DataPreprocessor
import settings
from anomaly_detector import AnomalyDetector


def main():
    preprocessor = DataPreprocessor(settings.file_path)
    detector = AnomalyDetector()
    df = preprocessor.read_data()
    data = preprocessor.fit_transform(df=df)
    outliers = detector.fit_predict(data)
    df = detector.add_outlier_column(data=df, outliers=outliers)
    df_no_outliers, df_with_outliers, indices_no_outliers, indices_with_outliers = detector.separate_data(data=df)
    transformed_columns = preprocessor.transform_columns(df_no_outliers, ["age", "balance"])

    print('yes')



if __name__ == "__main__":
    main()