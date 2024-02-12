from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, PowerTransformer
import pandas as pd
from sentence_transformers import SentenceTransformer
import os

class DataPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.pipeline = self.create_pipeline()
        self.pipe = Pipeline([('scaler', PowerTransformer())])

    def read_data(self):
        df = pd.read_csv(self.file_path, sep=";")
        return df.iloc[:, :8]

    def create_pipeline(self):
        categorical_transformer_onehot = Pipeline(
            steps=[
                ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False))
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

    def transform_columns(self, df, columns):

        return pd.DataFrame(self.pipe.fit_transform(df[columns]), columns=columns)

    def integrate_transformed_columns(self, original_df, transformed_df, columns):
        df_no_outliers_norm = original_df.copy()
        df_no_outliers_norm = df_no_outliers_norm.drop(columns, axis=1)

        for col in columns:
            df_no_outliers_norm[col] = transformed_df[col].values

        return df_no_outliers_norm

    def identify_and_index_categorical_columns(self, df):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

        categorical_columns = df.select_dtypes(exclude=numerics).columns

        categorical_columns_index = [df.columns.get_loc(col) for col in categorical_columns]

        return categorical_columns, categorical_columns_index

    def compile_and_encode_texts(self, df):
        def compile_text(x):
            text = f"""Age: {x['age']},  
                        housing loan: {x['housing']}, 
                        Job: {x['job']}, 
                        Marital: {x['marital']}, 
                        Education: {x['education']}, 
                        Default: {x['default']}, 
                        Balance: {x['balance']}, 
                        Personal loan: {x['loan']}
                    """
            return text

        sentences = df.apply(lambda x: compile_text(x), axis=1).tolist()
        model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")
        output = model.encode(sentences=sentences, show_progress_bar=True, normalize_embeddings=True)

        return pd.DataFrame(output)





