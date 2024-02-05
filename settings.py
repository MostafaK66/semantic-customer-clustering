file_path = 'data/train_data.csv'
K_RANGE = (2, 5)
K2_RANGE = range(2, 5)
N_INIT = 10
MAX_ITER = 100
INIT = 'k-means++'
CONTAMINATION = 0.001
RANDOM_STATE = 123
N_COMPONENTS = 3
PCA_N_ITER = 3
TSNE_LEARNING_RATE = 500
TSNE_PERPLEXITY = 200
TSNE_N_ITER = 5000


"""

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

categorical_columns = df_no_outliers_norm.select_dtypes(exclude=numerics).columns
print(categorical_columns)
categorical_columns_index = [df_no_outliers_norm.columns.get_loc(col) for col in categorical_columns]

"""