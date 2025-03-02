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
            # Fallback to a default style if seaborn style is not available
            plt.style.use('default')
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
            ax = sns.barplot(data=results_df, x='function_type', y='total_emissions', hue='function_type', palette=self.colors[:4], legend=False)
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
    
    def plot_industry_comparison(self, results: Dict[str, dict]) -> None:
        """Plot comparison of different industry scenarios.
        
        Args:
            results: Dictionary containing results for different industries
                    with metrics like emissions, costs, and production levels
        """
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot emissions comparison
            emissions = [data['total_emissions'] for data in results.values()]
            industries = list(results.keys())
            ax1.bar(industries, emissions, color=self.colors[:len(industries)])
            ax1.set_title('Total Emissions by Industry')
            ax1.set_ylabel('Total Emissions (tons)')
            ax1.grid(True, alpha=0.3)
            
            # Plot cost breakdown
            costs = np.array([
                [data['emission_cost'], data['production_cost']]
                for data in results.values()
            ])
            bottom = np.zeros(len(industries))
            
            for i, cost_type in enumerate(['Emission Cost', 'Production Cost']):
                ax2.bar(industries, costs[:, i], bottom=bottom,
                        label=cost_type, color=self.colors[i+2])
                bottom += costs[:, i]
            
            ax2.set_title('Cost Breakdown by Industry')
            ax2.set_ylabel('Cost ($)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(IMAGES_DIR, 'industry_comparison.pdf'),
                        dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error in plot_industry_comparison: {str(e)}")

    def plot_uncertainty_analysis(self, results_df: pd.DataFrame) -> None:
        """
        Plot the impact of demand uncertainty on emission patterns.

        Args:
            results_df: DataFrame with columns 'uncertainty', 'function_type',
                        'expected_cost', 'expected_emissions'.
        """
        try:
            plt.figure(figsize=(12, 8))
            
            # Get unique function types and create color palette
            func_types = results_df['function_type'].unique()
            palette = sns.color_palette('husl', n_colors=len(func_types))
            
            # Create line plot for each function type
            for i, func_type in enumerate(func_types):
                data = results_df[results_df['function_type'] == func_type]
                plt.plot(data['uncertainty'], data['expected_cost'],
                         marker='o', label=func_type, color=palette[i],
                         linewidth=2, markersize=8)
            
            plt.title('Impact of Demand Uncertainty on Expected Cost', fontsize=16)
            plt.xlabel('Demand Uncertainty (Coefficient of Variation)', fontsize=14)
            plt.ylabel('Expected Cost ($)', fontsize=14)
            plt.legend(title='Emission Function', fontsize=10, title_fontsize=12)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(IMAGES_DIR, 'demand_uncertainty_impact.pdf'), dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"Error in plot_uncertainty_analysis: {e}")

    def plot_sustainability_tradeoffs(self, results_df: pd.DataFrame) -> None:
        """
        Plot trade-offs between economic and environmental objectives.

        Args:
            results_df: DataFrame containing sustainability analysis data with
                        'emission_cost', 'emission_cap', 'total_cost',
                        'total_emissions', 'service_level', and 'function_type'
        """
        try:
            # Create a colormap for emission costs
            emission_costs = sorted(results_df['emission_cost'].unique())
            cmap = plt.get_cmap('viridis', len(emission_costs))

            # Scatter plot for Total Cost vs. Total Emissions, colored by Emission Cost
            plt.figure(figsize=(12, 8))
            for i, cost in enumerate(emission_costs):
                subset = results_df[results_df['emission_cost'] == cost]
                plt.scatter(subset['total_emissions'], subset['total_cost'],
                            color=cmap(i), label=f'Emission Cost: {cost}',
                            s=100, alpha=0.7, edgecolors='w')

            plt.title('Sustainability Trade-offs: Total Cost vs. Total Emissions', fontsize=16)
            plt.xlabel('Total Emissions (tons)', fontsize=14)
            plt.ylabel('Total Cost ($)', fontsize=14)
            plt.legend(title='Emission Cost ($/ton)', fontsize=10, title_fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(os.path.join(IMAGES_DIR, 'sustainability_tradeoffs.pdf'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error in plot_sustainability_tradeoffs: {str(e)}")
    def plot_benchmark_comparison(self, results_df: pd.DataFrame) -> None:
        """Plot benchmark comparison of nonlinear emission functions against linear baseline.
        
        Args:
            results_df: DataFrame containing benchmark comparison data
        """
        try:
            plt.figure(figsize=(12, 8))
            
            # Create grouped bar chart for cost and emission reductions
            x = np.arange(len(results_df['function_type'].unique()))
            width = 0.35
            
            # Group by function type and calculate means
            grouped = results_df.groupby('function_type').mean().reset_index()
            
            # Plot cost reduction
            ax1 = plt.subplot(2, 1, 1)
            bars1 = ax1.bar(x - width/2, grouped['cost_reduction_percent'], width, label='Cost Reduction', color=self.colors[0])
            ax1.set_ylabel('Cost Reduction (%)')
            ax1.set_title('Cost and Emission Reductions Compared to Linear Baseline')
            ax1.set_xticks(x)
            ax1.set_xticklabels(grouped['function_type'])
            ax1.grid(True, alpha=0.3)
            
            # Add value labels
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.1f}%', ha='center', va='bottom')
            
            # Plot emission reduction
            ax2 = plt.subplot(2, 1, 2)
            bars2 = ax2.bar(x - width/2, grouped['emission_reduction_percent'], width, label='Emission Reduction', color=self.colors[2])
            ax2.set_ylabel('Emission Reduction (%)')
            ax2.set_xlabel('Emission Function Type')
            ax2.set_xticks(x)
            ax2.set_xticklabels(grouped['function_type'])
            ax2.grid(True, alpha=0.3)
            
            # Add value labels
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.1f}%', ha='center', va='bottom')
            
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
            plt.figure(figsize=(10, 6))
            
            # Create line plot for each emission function type
            for i, func_type in enumerate(results_df['function_type'].unique()):
                data = results_df[results_df['function_type'] == func_type]
                plt.plot(data['problem_size'], data['solve_time'], marker='o', 
                         label=func_type, color=self.colors[i], linewidth=2)
            
            plt.title('Computational Performance Across Problem Sizes')
            plt.xlabel('Problem Size (Products × Periods)')
            plt.ylabel('Solution Time (seconds)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(IMAGES_DIR, 'computational_performance.pdf'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error in plot_computational_performance: {str(e)}")
    
    def plot_piecewise_analysis(self, results_df: pd.DataFrame) -> None:
        """Plot piecewise linear approximation analysis.
        
        Args:
            results_df: DataFrame containing piecewise approximation data
        """
        try:
            plt.figure(figsize=(12, 8))
            
            # Plot approximation error vs intervals
            ax1 = plt.subplot(2, 1, 1)
            for i, func_type in enumerate(results_df['function_type'].unique()):
                data = results_df[results_df['function_type'] == func_type]
                ax1.plot(data['intervals'], data['approximation_error'], marker='o', 
                        label=func_type, color=self.colors[i], linewidth=2)
            
            ax1.set_title('Piecewise Linear Approximation Analysis')
            ax1.set_ylabel('Approximation Error (%)')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Plot solution time vs intervals
            ax2 = plt.subplot(2, 1, 2)
            for i, func_type in enumerate(results_df['function_type'].unique()):
                data = results_df[results_df['function_type'] == func_type]
                ax2.plot(data['intervals'], data['solve_time'], marker='o', 
                        label=func_type, color=self.colors[i], linewidth=2)
            
            ax2.set_xlabel('Number of Piecewise Intervals (K)')
            ax2.set_ylabel('Solution Time (seconds)')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(IMAGES_DIR, 'piecewise_analysis.pdf'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error in plot_piecewise_analysis: {str(e)}")
    
    def plot_parameter_sensitivity(self, results_df: pd.DataFrame) -> None:
        """Plot sensitivity analysis for various model parameters.
        
        Args:
            results_df: DataFrame containing parameter sensitivity data
        """
        try:
            plt.figure(figsize=(14, 10))
            
            # Get unique parameters
            parameters = results_df['parameter'].unique()
            
            # Create subplots for each parameter
            for i, param in enumerate(parameters):
                param_data = results_df[results_df['parameter'] == param]
                
                # Plot cost sensitivity
                ax1 = plt.subplot(len(parameters), 2, 2*i+1)
                ax1.plot(param_data['value'], param_data['total_cost'], marker='o', 
                         color=self.colors[i], linewidth=2)
                ax1.set_title(f'{param} vs Total Cost')
                ax1.set_ylabel('Total Cost ($)')
                if i == len(parameters)-1:
                    ax1.set_xlabel(f'{param} Value')
                ax1.grid(True, alpha=0.3)
                
                # Plot emissions sensitivity
                ax2 = plt.subplot(len(parameters), 2, 2*i+2)
                ax2.plot(param_data['value'], param_data['total_emissions'], marker='o', 
                         color=self.colors[i+4], linewidth=2)
                ax2.set_title(f'{param} vs Total Emissions')
                ax2.set_ylabel('Total Emissions (tons)')
                if i == len(parameters)-1:
                    ax2.set_xlabel(f'{param} Value')
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(IMAGES_DIR, 'parameter_sensitivity.pdf'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error in plot_parameter_sensitivity: {str(e)}")
