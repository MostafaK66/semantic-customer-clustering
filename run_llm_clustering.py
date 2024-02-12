from utils import DataPreprocessor
import time
import settings

def main():
    start_time = time.time()
    preprocessor = DataPreprocessor(settings.file_path)
    df = preprocessor.read_data()
    df_embedding = preprocessor.compile_and_encode_texts(df=df)

    end_time = time.time()
    total_time_minutes = (end_time - start_time) / 60
    print(f"Total running time for LLM-Kmeans Clustering: {total_time_minutes:.2f} minutes")


if __name__ == "__main__":
    main()