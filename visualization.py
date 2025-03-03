import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from typing import Dict, List, Union
from config import IMAGES_DIR

class Visualizer:
    def __init__(self):
        """Initialize the visualizer with seaborn style and create images directory if needed."""
        try:
            plt.style.use('seaborn')
        except OSError:
            # Fallback to a standard matplotlib style if seaborn style is not available
            plt.style.use('ggplot')
        os.makedirs(IMAGES_DIR, exist_ok=True)
        self.colors = sns.color_palette('husl', 8)
        
    def plot_runtime_analysis(self, results_df: pd.DataFrame, x_var: str, title: str) -> None:
        """Plot runtime analysis against a specified variable.
        
        Args:
            results_df: DataFrame containing runtime data
            x_var: Variable to plot on x-axis
            title: Title for the plot
        """
        try:
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=results_df, x=x_var, y='runtime', marker='o', color=self.colors[0])
            plt.title(f'Runtime Analysis: {title}')
            plt.xlabel(title)
            plt.ylabel('Runtime (seconds)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(IMAGES_DIR, f'runtime_vs_{x_var.lower()}.pdf'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error in plot_runtime_analysis: {str(e)}")
    
    def plot_emission_comparison(self, results_df: pd.DataFrame) -> None:
        """Plot emission comparison across different function types.
        
        Args:
            results_df: DataFrame containing emission data
        """
        try:
            plt.figure(figsize=(12, 6))
            ax = sns.barplot(data=results_df, x='function_type', y='total_emissions', palette=self.colors)
            plt.title('Emission Comparison Across Different Functions', pad=20)
            plt.xlabel('Emission Function Type')
            plt.ylabel('Total Emissions (tons)')
            plt.xticks(rotation=45)
            
            # Add value labels on top of bars
            for i, bar in enumerate(ax.patches):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{bar.get_height():.1f}',
                        ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(IMAGES_DIR, 'emission_comparison.pdf'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error in plot_emission_comparison: {str(e)}")
    
    def plot_sensitivity_analysis(self, results_df: pd.DataFrame, x_var: str, y_var: str, title: str) -> None:
        """Plot sensitivity analysis for different function types.
        
        Args:
            results_df: DataFrame containing sensitivity analysis data
            x_var: Variable for x-axis
            y_var: Variable for y-axis
            title: Title for the plot
        """
        try:
            plt.figure(figsize=(10, 6))
            for i, func_type in enumerate(results_df['function_type'].unique()):
                data = results_df[results_df['function_type'] == func_type]
                plt.plot(data[x_var], data[y_var], marker='o', label=func_type,
                         color=self.colors[i], linewidth=2, markersize=8)
            
            plt.title(title, pad=20)
            plt.xlabel(x_var)
            plt.ylabel(y_var)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(IMAGES_DIR, f'sensitivity_{y_var.lower()}.pdf'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error in plot_sensitivity_analysis: {str(e)}")
    
    def plot_production_pattern(self, production_data: pd.DataFrame, scenario: int) -> None:
        """Plot production patterns for different products in a given scenario.
        
        Args:
            production_data: DataFrame containing production quantities
            scenario: Scenario number to plot
        """
        try:
            plt.figure(figsize=(12, 6))
            for i in range(production_data.shape[1]):
                plt.plot(range(len(production_data)), production_data.iloc[:, i],
                         marker='o', label=f'Product {i+1}', color=self.colors[i],
                         linewidth=2, markersize=6)
            
            plt.title(f'Production Pattern - Scenario {scenario}', pad=20)
            plt.xlabel('Time Period')
            plt.ylabel('Production Quantity')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(IMAGES_DIR, f'production_pattern_s{scenario}.pdf'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error in plot_production_pattern: {str(e)}")
    
    def plot_inventory_levels(self, inventory_data: pd.DataFrame, scenario: int) -> None:
        """Plot inventory levels for different products in a given scenario.
        
        Args:
            inventory_data: DataFrame containing inventory levels
            scenario: Scenario number to plot
        """
        try:
            plt.figure(figsize=(12, 6))
            for i in range(inventory_data.shape[1]):
                plt.plot(range(len(inventory_data)), inventory_data.iloc[:, i],
                         marker='s', label=f'Product {i+1}', color=self.colors[i],
                         linewidth=2, markersize=6)
            
            plt.title(f'Inventory Levels - Scenario {scenario}', pad=20)
            plt.xlabel('Time Period')
            plt.ylabel('Inventory Level')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(IMAGES_DIR, f'inventory_levels_s{scenario}.pdf'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error in plot_inventory_levels: {str(e)}")