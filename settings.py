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

pipe = Pipeline([('scaler', PowerTransformer())])

df_aux = pd.DataFrame(pipe_fit.fit_transform(df_no_outliers[["age", "balance"]] ), columns = ["age", "balance"])
df_no_outliers_norm = df_no_outliers.copy()

# Replace age and balance columns by preprocessed values
df_no_outliers_norm = df_no_outliers_norm.drop(["age", "balance"], axis = 1)
df_no_outliers_norm["age"] = df_aux["age"].values
df_no_outliers_norm["balance"] = df_aux["balance"].values
df_no_outliers_norm

"""