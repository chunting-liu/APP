import numpy as np
import pandas as pd
import time
from app_model import APPModel
from visualization import Visualizer

class ExperimentRunner:
    def __init__(self):
        self.visualizer = Visualizer()
        
    def run_emission_pattern_analysis(self):
        """Analyze how different emission patterns affect production decisions"""
        model = APPModel()
        results = []
        
        # Test different emission functions across production volumes
        production_levels = np.linspace(50, 250, 20)  # From min to capacity
        emission_types = ['linear', 'quadratic', 'exponential', 'logarithmic']
        
        for func_type in emission_types:
            total_cost, total_emissions, service_level, avg_inventory = model.solve(
                emission_type=func_type,
                production_levels=production_levels
            )
            results.append({
                'function_type': func_type,
                'total_emissions': total_emissions,
                'total_cost': total_cost,
            })
        
        self.visualizer.plot_emission_comparison(pd.DataFrame(results))
        return pd.DataFrame(results)
    def run_industry_case_studies(self):
        """Run case studies for steel and semiconductor industries"""
        # Steel industry parameters
        steel_params = {
            'alpha': 0.15,  # Base emission factor
            'beta': 0.003,  # Quadratic term for increasing emissions
            'capacity': 200,  # Units per period
            'demand_mean': 150,
            'demand_std': 30
        }
        
        # Semiconductor industry parameters
        semi_params = {
            'alpha': 2.0,  # Scaling factor
            'beta': 0.2,   # Rate parameter
            'capacity': 300,
            'demand_mean': 200,
            'demand_std': 40
        }
        
        results = {
            'steel': self.run_industry_scenario('steel', steel_params, 'quadratic'),  # Higher emissions at high production
            'semi': self.run_industry_scenario('semiconductor', semi_params, 'logarithmic')  # Efficiency gains at scale
        }
        
        self.visualizer.plot_industry_comparison(results)
        return results

    def run_sustainability_analysis(self):
        """Analyze trade-offs between economic and environmental objectives"""
        emission_costs = [20, 50, 80]  # $/ton
        emission_caps = [1500, 2000, 2500]  # tons/period
        results = []
        
        for cost in emission_costs:
            for cap in emission_caps:
                for func_type in ['linear', 'quadratic', 'exponential', 'logarithmic']:
                    model = APPModel(emission_cost=cost, emission_cap=cap)
                    total_cost, total_emissions, service_level, avg_inventory = model.solve(emission_type=func_type)
                    
                    results.append({
                        'emission_cost': cost,
                        'emission_cap': cap,
                        'function_type': func_type,
                        'total_cost': total_cost,
                        'total_emissions': total_emissions,
                        'service_level': service_level,
                        'inventory_levels': avg_inventory
                    })
        
        self.visualizer.plot_sustainability_tradeoffs(pd.DataFrame(results))
        return pd.DataFrame(results)

    def run_industry_scenario(self, industry_type, params, emission_type):
        """Run specific industry scenario"""
        # Extract model parameters
        model_params = {
            'emission_cap': params.get('emission_cap', 2500),
            'demand_uncertainty': params.get('demand_uncertainty', None)
        }
        
        # Create model with proper parameters
        model = APPModel(**model_params)
        
        # Solve the model with the specified emission type
        return model.solve(emission_type=emission_type)

    def analyze_demand_uncertainty(self):
        """Analyze impact of demand uncertainty on emission patterns"""
        # Pass emission parameters to solve method
        emission_params = {
            'alpha': params.get('alpha'),
            'beta': params.get('beta')
        }
        return model.solve(emission_type=emission_type, **emission_params)
        results = []
        
        for uncertainty in uncertainty_levels:
            model = APPModel(demand_uncertainty=uncertainty)
            for func_type in ['linear', 'quadratic', 'exponential', 'logarithmic']:
                total_cost, total_emissions, service_level, avg_inventory = model.solve(emission_type=func_type)
                results.append({
                    'uncertainty': uncertainty,
                    'function_type': func_type,
                    'expected_cost': total_cost,
                    'expected_emissions': total_emissions,
                    'service_level': service_level,
                    'avg_inventory': avg_inventory
                })
        self.visualizer.plot_uncertainty_analysis(pd.DataFrame(results))
        return pd.DataFrame(results)

    def run_sensitivity_analysis(self):
        """Perform sensitivity analysis"""
        emission_costs = [20, 40, 60, 80]
        results = []
        
        for cost in emission_costs:
            model = APPModel(emission_cost=cost)
            for func_type in ['linear', 'quadratic', 'exponential', 'logarithmic']:
                total_cost, total_emissions = model.solve(emission_type=func_type)
                results.append({
                    'emission_cost': cost,
                    'function_type': func_type,
                    'total_cost': total_cost,
                    'total_emissions': total_emissions
                })
        
        results_df = pd.DataFrame(results)
        self.visualizer.plot_sensitivity_analysis(
            results_df, 
            'emission_cost', 
            'total_cost', 
            'Sensitivity to Emission Costs'
        )
    def run_benchmark_comparison(self):
        """Compare performance against traditional linear APP model"""
        problem_sizes = [(5, 12), (10, 24), (15, 36)]  # (products, periods)
        results = []
        
        for num_products, num_periods in problem_sizes:
            # Traditional linear model
            linear_model = APPModel()
            linear_model.I = num_products
            linear_model.T = num_periods
            linear_model._generate_parameters()
            linear_total_cost, linear_total_emissions, _, _ = linear_model.solve(emission_type='linear')
            
            # Nonlinear models
            for func_type in ['quadratic', 'exponential', 'logarithmic']:
                model = APPModel()
                model.I = num_products
                model.T = num_periods
                model._generate_parameters()
                total_cost, total_emissions, _, _ = model.solve(emission_type=func_type)
                
                # Calculate improvements
                cost_reduction = ((linear_total_cost - total_cost) 
                                / linear_total_cost * 100)
                emission_reduction = ((linear_total_emissions - total_emissions) 
                                    / linear_total_emissions * 100)
                
                results.append({
                    'products': num_products,
                    'periods': num_periods,
                    'function_type': func_type,
                    'cost_reduction_percent': cost_reduction,
                    'emission_reduction_percent': emission_reduction,
                    'computation_time': 0  # Remove metrics['solve_time'] as it's not returned by solve()
                })
        
        self.visualizer.plot_benchmark_comparison(pd.DataFrame(results))
        return pd.DataFrame(results)
    
    def analyze_computational_performance(self):
        """Analyze computational performance across different problem sizes"""
        problem_sizes = [(5, 12), (10, 24), (15, 36), (20, 48)]
        results = []
        
        for num_products, num_periods in problem_sizes:
            for func_type in ['linear', 'quadratic', 'exponential', 'logarithmic']:
                model = APPModel()
                model.I = num_products
                model.T = num_periods
                model._generate_parameters()
                
                # Measure solve time
                start_time = time.time()
                total_cost, total_emissions, service_level, avg_inventory = model.solve(emission_type=func_type)
                solve_time = time.time() - start_time
                
                results.append({
                    'products': num_products,
                    'periods': num_periods,
                    'function_type': func_type,
                    'problem_size': num_products * num_periods,
                    'solve_time': solve_time,
                    'total_cost': total_cost,
                    'total_emissions': total_emissions
                })
        
        self.visualizer.plot_computational_performance(pd.DataFrame(results))
        return pd.DataFrame(results)
    
    def analyze_piecewise_approximation(self):
        """Analyze impact of number of piecewise intervals"""
        K_values = [3, 5, 7, 10, 15]
        results = []
        
        # Reference solution with high K value for error calculation
        reference_K = 30
        reference_results = {}
        
        # Get reference solutions for each function type
        for func_type in ['quadratic', 'exponential', 'logarithmic']:
            ref_model = APPModel()
            ref_model.K = reference_K
            ref_model._generate_parameters()
            ref_cost, ref_emissions, _, _ = ref_model.solve(emission_type=func_type)
            reference_results[func_type] = {'cost': ref_cost, 'emissions': ref_emissions}
        
        for K in K_values:
            for func_type in ['quadratic', 'exponential', 'logarithmic']:
                model = APPModel()
                model.K = K
                model._generate_parameters()
                
                # Measure solve time
                start_time = time.time()
                total_cost, total_emissions, _, _ = model.solve(emission_type=func_type)
                solve_time = time.time() - start_time
                
                # Calculate approximation error compared to reference solution
                ref_cost = reference_results[func_type]['cost']
                ref_emissions = reference_results[func_type]['emissions']
                
                cost_error = abs((total_cost - ref_cost) / ref_cost * 100) if ref_cost != 0 else 0
                emission_error = abs((total_emissions - ref_emissions) / ref_emissions * 100) if ref_emissions != 0 else 0
                approximation_error = (cost_error + emission_error) / 2
                
                results.append({
                    'intervals': K,
                    'function_type': func_type,
                    'approximation_error': approximation_error,
                    'solve_time': solve_time,
                    'total_cost': total_cost,
                    'total_emissions': total_emissions
                })
        
        self.visualizer.plot_piecewise_analysis(pd.DataFrame(results))
        return pd.DataFrame(results)
    
    def run_parameter_sensitivity(self):
        """Analyze sensitivity to various model parameters"""
        parameters = {
            'capacity': [150, 200, 250, 300],
            'backorder_cost': [150, 200, 250, 300],
            'emission_alpha': [0.1, 0.2, 0.3, 0.4],
            'emission_beta': [0.01, 0.02, 0.03, 0.04]
        }
        
        results = []
        base_model = APPModel()
        
        for param_name, param_values in parameters.items():
            for value in param_values:
                # Adjust emission cap based on parameter values
                emission_cap = 5000
                if param_name in ['emission_alpha', 'emission_beta']:
                    # Increase emission cap proportionally for higher emission parameters
                    emission_cap = 5000 * (1 + value)
                
                model = APPModel(emission_cap=emission_cap)
                
                # Update the specific parameter
                if param_name == 'capacity':
                    model.cap = np.full((model.I, model.T), value)
                elif param_name == 'backorder_cost':
                    model.c_b = np.full(model.I, value)
                elif param_name == 'emission_alpha':
                    model.alpha_quad = np.full(model.I, value)
                elif param_name == 'emission_beta':
                    model.beta_quad = np.full(model.I, value)
                
                total_cost, total_emissions, service_level, avg_inventory = model.solve(emission_type='quadratic')
                
                results.append({
                    'parameter': param_name,
                    'value': value,
                    'total_cost': total_cost,
                    'total_emissions': total_emissions,
                    'service_level': service_level
                })
        
        self.visualizer.plot_parameter_sensitivity(pd.DataFrame(results))
        return pd.DataFrame(results)
