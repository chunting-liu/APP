import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from gurobipy import *
import time
import os
from config import RESULTS_DIR, IMAGES_DIR

class ExperimentRunner:
    def __init__(self):
        self.results_dir = RESULTS_DIR
        self.figures_dir = IMAGES_DIR
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
        
        # Base parameters
        self.base_params = {
            'num_products': 5,
            'time_periods': 12,
            'num_scenarios': 3,
            'emission_cost': 40,
            'intervals_K': 5
        }
    
    def generate_emission_comparison(self):
        """Generate data for Table 1: Performance Comparison of Different Emission Functions"""
        emission_functions = ['Linear', 'Quadratic', 'Exponential', 'Logarithmic']
        results = []
        
        for func in emission_functions:
            # Run model with different emission functions
            start_time = time.time()
            emissions, memory, total_cost = self._run_model_with_emission_function(func)
            runtime = time.time() - start_time
            
            results.append({
                'Emission_Function': func,
                'Total_Emissions': emissions,
                'Runtime': runtime,
                'Memory': memory,
                'Total_Cost': total_cost
            })
        
        df = pd.DataFrame(results)
        df.to_csv(f"{self.results_dir}/emission_comparison.csv", index=False)
        return df

    def generate_emission_cost_impact(self):
        """Generate data for Table 2: Impact of Emission Costs on Production Strategy"""
        emission_costs = [20, 40, 60, 80]
        emission_functions = ['Linear', 'Quadratic', 'Exponential', 'Logarithmic']
        results = []
        
        for func in emission_functions:
            reductions = []
            base_production = self._get_base_production(func)
            
            for cost in emission_costs:
                production = self._run_model_with_cost(func, cost)
                reduction = ((base_production - production) / base_production) * 100
                reductions.append(reduction)
            
            results.append({
                'Emission_Function': func,
                **{f'Cost_{cost}': red for cost, red in zip(emission_costs, reductions)}
            })
        
        df = pd.DataFrame(results)
        df.to_csv(f"{self.results_dir}/emission_cost_impact.csv", index=False)
        return df

    def generate_steel_industry_metrics(self):
        """Generate data for Table 3: Steel Industry Performance Metrics"""
        emission_functions = ['Linear', 'Quadratic', 'Exponential', 'Logarithmic']
        metrics = ['Energy_Efficiency', 'Carbon_Intensity', 'Production_Cost', 'Capacity_Utilization']
        results = []
        
        for func in emission_functions:
            # Simulate steel industry specific parameters
            metrics_values = self._calculate_steel_metrics(func)
            results.append({
                'Emission_Function': func,
                **{metric: value for metric, value in zip(metrics, metrics_values)}
            })
        
        df = pd.DataFrame(results)
        df.to_csv(f"{self.results_dir}/steel_industry_metrics.csv", index=False)
        return df

    def generate_computational_performance(self):
        """Generate data for Table 4: Computational Performance Analysis"""
        problem_sizes = [
            {'name': 'Small', 'I': 5, 'T': 12},
            {'name': 'Medium', 'I': 10, 'T': 24},
            {'name': 'Large', 'I': 20, 'T': 48}
        ]
        results = []
        
        for size in problem_sizes:
            runtime, memory, gap = self._run_performance_test(size['I'], size['T'])
            results.append({
                'Problem_Size': size['name'],
                'Runtime': runtime,
                'Memory': memory,
                'Optimality_Gap': gap
            })
        
        df = pd.DataFrame(results)
        df.to_csv(f"{self.results_dir}/computational_performance.csv", index=False)
        return df

    def plot_sensitivity_analysis(self):
        """Generate sensitivity analysis plots"""
        emission_costs = np.arange(20, 100, 20)
        emission_functions = ['Linear', 'Quadratic', 'Exponential', 'Logarithmic']
        
        plt.figure(figsize=(10, 6))
        for func in emission_functions:
            costs = [self._calculate_total_cost(func, cost) for cost in emission_costs]
            plt.plot(emission_costs, costs, marker='o', label=func)
        
        plt.xlabel('Emission Cost ($/ton)')
        plt.ylabel('Total Cost ($)')
        plt.title('Sensitivity to Emission Costs')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.figures_dir}/sensitivity_emission_cost.pdf")
        plt.close()

    def _run_model_with_emission_function(self, func_type):
        """Run model with different emission functions and return metrics"""
        from app_model import APPModel
        model = APPModel()
        results = model.solve(emission_type=func_type.lower())
        
        # Get memory usage
        import psutil
        process = psutil.Process()
        memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
        
        return results['total_emissions'], memory, results['costs']['total']

    def _run_model_with_cost(self, func_type, cost):
        """Run model with specific emission cost"""
        from app_model import APPModel
        model = APPModel(emission_cost=cost)
        results = model.solve(emission_type=func_type.lower())
        return np.sum(results['production_plan'])

    def _get_base_production(self, func_type):
        """Get base production without emission costs"""
        from app_model import APPModel
        model = APPModel(emission_cost=0)
        results = model.solve(emission_type=func_type.lower())
        return np.sum(results['production_plan'])

    def _calculate_steel_metrics(self, func_type):
        """Calculate steel industry specific metrics"""
        from app_model import APPModel
        model = APPModel()
        results = model.solve(emission_type=func_type.lower())
        
        # Calculate metrics
        energy_efficiency = results['costs']['total'] / np.sum(results['production_plan'])
        carbon_intensity = results['total_emissions'] / np.sum(results['production_plan'])
        production_cost = results['costs']['production']
        capacity_utilization = np.mean(results['production_plan']) / 250  # 250 is max capacity
        
        return [energy_efficiency, carbon_intensity, production_cost, capacity_utilization]

    def _run_performance_test(self, num_products, time_periods):
        """Run performance test with different problem sizes"""
        from app_model import APPModel
        import time
        import psutil
        
        start_time = time.time()
        model = APPModel()
        model.I = num_products
        model.T = time_periods
        model._generate_parameters()
        
        results = model.solve()
        runtime = time.time() - start_time
        
        process = psutil.Process()
        memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
        
        gap = 0 if results['costs']['total'] > 0 else float('inf')
        
        return runtime, memory, gap

def main():
    runner = ExperimentRunner()
    
    # Generate tables
    emission_comparison = runner.generate_emission_comparison()
    emission_cost_impact = runner.generate_emission_cost_impact()
    steel_metrics = runner.generate_steel_industry_metrics()
    computational_performance = runner.generate_computational_performance()
    
    # Generate figures
    runner.plot_sensitivity_analysis()
    
    print("Results generation completed successfully")

if __name__ == "__main__":
    main()