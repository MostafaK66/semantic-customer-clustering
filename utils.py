from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, PowerTransformer
import pandas as pd


class DataPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.pipeline = self.create_pipeline()

    def read_data(self):
        df = pd.read_csv(self.file_path, sep=";")
        return df.iloc[:, :8]

    def create_pipeline(self):
        categorical_transformer_onehot = Pipeline(
            steps=[
                ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first", sparse=False))
            ])

        categorical_transformer_ordinal = Pipeline(
            steps=[
                ("encoder", OrdinalEncoder())
            ])

        num = Pipeline(
            steps=[
                ("encoder", PowerTransformer())
            ])

        preprocessor = ColumnTransformer(transformers=[
            ('cat_onehot', categorical_transformer_onehot, ["default", "housing", "loan", "job", "marital"]),
            ('cat_ordinal', categorical_transformer_ordinal, ["education"]),
            ('num', num, ["age", "balance"])
        ])

        return Pipeline(steps=[("preprocessor", preprocessor)])

    def fit_transform(self, df):
        self.pipeline.fit(df)
        transformed_data = self.pipeline.transform(df)
        columns = self.pipeline.get_feature_names_out().tolist()
        return pd.DataFrame(transformed_data, columns=columns)

