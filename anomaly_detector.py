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
        data_with_outliers = data.drop(["outliers"], axis=1)
        return data_no_outliers, data_with_outliers

