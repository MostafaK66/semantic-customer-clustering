import matplotlib.pyplot as plt
import numpy as np
import os
import plotly.express as px

class DataPlotter:
    def __init__(self):
        self.output_dir = 'output'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def plot_ecdf(self, data, columns):
        fig, axs = plt.subplots(len(columns), 1, figsize=(8, 5*len(columns)))

        for idx, column in enumerate(columns):
            normal_data = data[data['outliers'] == 0][column]
            outlier_data = data[data['outliers'] == 1][column]

            x_normal = np.sort(normal_data)
            y_normal = np.arange(1, len(x_normal) + 1) / len(normal_data)
            axs[idx].plot(x_normal, y_normal, marker='.', linestyle='none', color='blue', label='Normal')

            x_outlier = np.sort(outlier_data)
            y_outlier = np.arange(1, len(x_outlier) + 1) / len(outlier_data)
            axs[idx].plot(x_outlier, y_outlier, marker='.', linestyle='none', color='red', label='Outlier')

            axs[idx].set_title(f'ECDF of {column}')
            axs[idx].set_xlabel(column)
            axs[idx].set_ylabel('ECDF')
            axs[idx].grid(True)
            axs[idx].legend()

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/ecdf_subplots_outliers.png')
        plt.close()

    def plot_3d(self, df, title, opacity=0.8, width_line=0.1):
        df = df.astype({"cluster": "object"})
        df = df.sort_values("cluster")

        columns = df.columns[0:3].tolist()

        fig = px.scatter_3d(df,
                            x=columns[0],
                            y=columns[1],
                            z=columns[2],
                            color='cluster',
                            template="plotly",
                            color_discrete_sequence=px.colors.qualitative.Vivid,
                            title=title).update_traces(
            marker={
                "size": 4,
                "opacity": opacity,
                "line": {
                    "width": width_line,
                    "color": "black",
                }
            }
        ).update_layout(
            width=1000,
            height=800,
            autosize=False,
            showlegend=True,
            legend=dict(title_font_family="Times New Roman",
                        font=dict(size=20)),
            scene=dict(xaxis=dict(title='comp1', titlefont_color='black'),
                       yaxis=dict(title='comp2', titlefont_color='black'),
                       zaxis=dict(title='comp3', titlefont_color='black')),
            font=dict(family="Gilroy", color='black', size=15))

        fig.show()

    # def plot_clustering_cost(self, df_cost):
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(df_cost['Cluster'], df_cost['Cost'], marker='o')
    #     plt.title('Clustering Cost vs Number of Clusters')
    #     plt.xlabel('Number of Clusters')
    #     plt.ylabel('Cost')
    #     plt.grid(True)
    #     plt.savefig(os.path.join(self.output_dir, 'clustering_cost.png'))
    #     plt.close()