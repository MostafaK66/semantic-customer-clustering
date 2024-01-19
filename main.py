from utils import DataPreprocessor
import settings


def main():
    preprocessor = DataPreprocessor(settings.file_path)
    data = preprocessor.read_data()
    data_transformed = preprocessor.fit_transform(df=data)
    print(data_transformed)


if __name__ == "__main__":
    main()


