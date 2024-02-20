import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import os


class FeatureImportanceAnalyzer:
    def __init__(self, colsample_bytree=0.8):
        self.colsample_by_tree = colsample_bytree
        self.clf = None
        self.explainer = None
        self.shap_values = None

    def generate_shap_summary(self, filtered_df, clusters_predict):
        self.clf = lgb.LGBMClassifier(colsample_by_tree=self.colsample_by_tree)

        for col in ["job", "marital", "education", "housing", "loan", "default"]:
            filtered_df[col] = filtered_df[col].astype('category')

        self.clf.fit(X=filtered_df, y=clusters_predict)

        self.explainer = shap.TreeExplainer(self.clf)
        self.shap_values = self.explainer.shap_values(filtered_df)

        shap.summary_plot(self.shap_values, filtered_df, plot_type="bar", show=False)

        output_dir = 'output'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        plt.gcf().set_size_inches(15, 10)
        plt.savefig(os.path.join(output_dir, 'shap_summary_plot.png'))
        plt.close()


