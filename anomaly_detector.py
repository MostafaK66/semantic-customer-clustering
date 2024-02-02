from pyod.models.ecod import ECOD
import settings

class AnomalyDetector:
    def __init__(self):
        self.clf = ECOD(contamination=settings.CONTAMINATION)

    def fit_predict(self, data):
        self.clf.fit(data)
        return self.clf.predict(data)

    def add_outlier_column(self, data, outliers):
        data["outliers"] = outliers
        return data

    def separate_data(self, data):
        data_no_outliers = data[data["outliers"] == 0].drop(["outliers"], axis=1)

        data_with_outliers = data[data["outliers"] == 1].drop(["outliers"], axis=1)

        indices_no_outliers = data[data["outliers"] == 0].index.to_frame(name='index_no_outliers')

        indices_with_outliers = data[data["outliers"] == 1].index.to_frame(name='index_with_outliers')

        return data_no_outliers, data_with_outliers, indices_no_outliers, indices_with_outliers

    def filter_outliers(self, df, index_with_outliers):
        outlier_indices = index_with_outliers['index_with_outliers'].tolist()

        return df[~df.index.isin(outlier_indices)]

    # Example usage:
    # filtered_df = filter_outliers(df, index_with_outliers)



