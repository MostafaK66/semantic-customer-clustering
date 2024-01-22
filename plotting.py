import matplotlib.pyplot as plt
import numpy as np
import os


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