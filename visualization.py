import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

class Visualizer:
    def __init__(self):
        plt.style.use('seaborn')
        
    def plot_runtime_analysis(self, results_df, x_var, title):
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=results_df, x=x_var, y='runtime', marker='o')
        plt.title(f'Runtime vs {title}')
        plt.xlabel(title)
        plt.ylabel('Runtime (seconds)')
        plt.grid(True)
        plt.savefig(f'c:/Users/group/Downloads/images/runtime_vs_{x_var.lower()}.pdf')
        plt.close()
    
    def plot_emission_comparison(self, results_df):
        plt.figure(figsize=(12, 6))
        sns.barplot(data=results_df, x='function_type', y='total_emissions')
        plt.title('Emission Comparison Across Different Functions')
        plt.xlabel('Emission Function Type')
        plt.ylabel('Total Emissions (tons)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('c:/Users/group/Downloads/images/emission_comparison.pdf')
        plt.close()
    
    def plot_sensitivity_analysis(self, results_df, x_var, y_var, title):
        plt.figure(figsize=(10, 6))
        for func_type in results_df['function_type'].unique():
            data = results_df[results_df['function_type'] == func_type]
            plt.plot(data[x_var], data[y_var], marker='o', label=func_type)
        
        plt.title(title)
        plt.xlabel(x_var)
        plt.ylabel(y_var)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'c:/Users/group/Downloads/images/sensitivity_{y_var.lower()}.pdf')
        plt.close()