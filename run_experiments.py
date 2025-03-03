import experiments
from experiments import ExperimentRunner
import logging
import os
import pandas as pd
import time
from config import LOG_FILE, RESULTS_DIR, IMAGES_DIR

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=LOG_FILE
    )

def main():
    setup_logging()
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    runner = experiments.ExperimentRunner()
    results = {}
    
    try:
        # Analyze emission patterns and their effects
        logging.info("Analyzing emission patterns...")
        results['emission_patterns'] = runner.run_emission_pattern_analysis()
        
        # Run industry case studies
        logging.info("Running industry case studies...")
        results['industry_cases'] = runner.run_industry_case_studies()
        
        # Analyze sustainability trade-offs
        logging.info("Analyzing sustainability trade-offs...")
        results['sustainability'] = runner.run_sustainability_analysis()
        
        # Analyze impact of demand uncertainty
        logging.info("Analyzing demand uncertainty effects...")
        results['uncertainty'] = runner.analyze_demand_uncertainty()
        
        # Run benchmark comparison against linear emission model
        logging.info("Running benchmark comparison against linear emission model...")
        results['benchmark_comparison'] = runner.run_benchmark_comparison()
        
        # Analyze computational performance across different problem sizes
        logging.info("Analyzing computational performance...")
        results['computational_performance'] = runner.analyze_computational_performance()
        
        # Analyze piecewise linear approximation
        logging.info("Analyzing piecewise linear approximation...")
        results['piecewise_analysis'] = runner.analyze_piecewise_approximation()
        
        # Run parameter sensitivity analysis
        logging.info("Running parameter sensitivity analysis...")
        results['parameter_sensitivity'] = runner.run_parameter_sensitivity()
        
        # Run sensitivity analysis
        logging.info("Running sensitivity analysis...")
        runner.run_sensitivity_analysis()
        
        # Save results
        for name, result in results.items():
            if isinstance(result, pd.DataFrame):
                result.to_csv(os.path.join(RESULTS_DIR, f'{name}_results.csv'))
            else:
                logging.info(f"Skipping saving {name} as it is not a DataFrame.")
        
        logging.info("All experiments completed successfully")
        
    except Exception as e:
        logging.error(f"Error during experiments: {str(e)}")
        raise

if __name__ == "__main__":
    main()
