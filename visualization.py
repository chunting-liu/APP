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

    def plot_sustainability_tradeoffs(self, results_df: pd.DataFrame) -> None:
        """Plot sustainability trade-offs between cost and emissions.
        
        Args:
            results_df: DataFrame containing sustainability analysis data
        """
        try:
            plt.figure(figsize=(12, 6))
            sns.scatterplot(data=results_df, x='total_cost', y='total_emissions', 
                            hue='function_type', size='emission_cap', sizes=(20, 200),
                            palette=self.colors[:len(results_df['function_type'].unique())])
            plt.title('Sustainability Trade-offs: Cost vs. Emissions', pad=20)
            plt.xlabel('Total Cost ($)')
            plt.ylabel('Total Emissions (tons)')
            plt.legend(title='Emission Function', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(IMAGES_DIR, 'sustainability_tradeoffs.pdf'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error in plot_sustainability_tradeoffs: {str(e)}")

    def plot_benchmark_comparison(self, results_df: pd.DataFrame) -> None:
        """Plot benchmark comparison results.
        
        Args:
            results_df: DataFrame containing benchmark comparison data
        """
        try:
            plt.figure(figsize=(12, 6))
            sns.barplot(data=results_df, x='function_type', y='cost_reduction_percent', palette=self.colors)
            plt.title('Benchmark Comparison: Cost Reduction', pad=20)
            plt.xlabel('Emission Function Type')
            plt.ylabel('Cost Reduction (%)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(IMAGES_DIR, 'benchmark_comparison.pdf'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error in plot_benchmark_comparison: {str(e)}")

    def plot_computational_performance(self, results_df: pd.DataFrame) -> None:
        """Plot computational performance across different problem sizes.
        
        Args:
            results_df: DataFrame containing computational performance data
        """
        try:
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=results_df, x='problem_size', y='solve_time', hue='function_type', palette=self.colors)
            plt.title('Computational Performance vs. Problem Size', pad=20)
            plt.xlabel('Problem Size (Products x Periods)')
            plt.ylabel('Solve Time (seconds)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(IMAGES_DIR, 'computational_performance.pdf'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error in plot_computational_performance: {str(e)}")


    def plot_parameter_sensitivity(self, df: pd.DataFrame):
        """
        Plots the parameter sensitivity data.

        Args:
            df: A pandas DataFrame containing the sensitivity results.
        """
        #Implement your plot code here
        print("plotted")

        # Example plot: Assumes columns 'parameter' and 'value' are in the dataframe
        if 'parameter' in df.columns and 'value' in df.columns:
            plt.figure()
            plt.plot(df['parameter'], df['value'])
            plt.xlabel('Parameter')
            plt.ylabel('Value')
            plt.title('Parameter Sensitivity')
            plt.savefig(os.path.join(IMAGES_DIR, 'paramter_sensitivity.pdf'), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            print("Missing parameter or value in DataFrame")
        pass

    def plot_sustainability_analysis(self, results_df: pd.DataFrame) -> None:
        """Plot sustainability analysis results as bar charts.
        
        Args:
            results_df: DataFrame containing sustainability analysis data
        """
        try:
            # Group data by emission function and emission cap
            grouped_data = results_df.groupby(['function_type', 'emission_cap'])
            
            # Prepare data for plotting
            function_types = results_df['function_type'].unique()
            emission_caps = results_df['emission_cap'].unique()
            
            # Set up the plot
            fig, axes = plt.subplots(len(emission_caps), 2, figsize=(15, 5 * len(emission_caps)))
            fig.suptitle('Sustainability Analysis: Cost and Emissions vs. Emission Cap and Function Type', fontsize=16)
            
            # Iterate through emission caps and create subplots
            for i, cap in enumerate(emission_caps):
                ax1 = axes[i, 0]
                ax2 = axes[i, 1]
                
                # Filter data for the current emission cap
                cap_data = results_df[results_df['emission_cap'] == cap]
                
                # Plot total cost
                sns.barplot(ax=ax1, x='function_type', y='total_cost', data=cap_data, palette=self.colors[:len(function_types)])
                ax1.set_title(f'Total Cost vs. Function Type (Cap = {cap})')
                ax1.set_xlabel('Emission Function Type')
                ax1.set_ylabel('Total Cost ($)')
                ax1.tick_params(axis='x', rotation=45)
                
                # Plot total emissions
                sns.barplot(ax=ax2, x='function_type', y='total_emissions', data=cap_data, palette=self.colors[:len(function_types)])
                ax2.set_title(f'Total Emissions vs. Function Type (Cap = {cap})')
                ax2.set_xlabel('Emission Function Type')
                ax2.set_ylabel('Total Emissions (tons)')
                ax2.tick_params(axis='x', rotation=45)
            
            # Adjust layout and save the figure
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(IMAGES_DIR, 'sustainability_analysis.pdf'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error in plot_sustainability_analysis: {str(e)}")
