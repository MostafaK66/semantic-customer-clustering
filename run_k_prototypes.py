from utils import DataPreprocessor
import settings



def main():
    preprocessor = DataPreprocessor(settings.file_path)
    df = preprocessor.read_data()
    data_norm = preprocessor.encode_and_transform(df=df)

    print('yes')



if __name__ == "__main__":
    main()