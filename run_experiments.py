from experiments import ExperimentRunner
import logging
import os
import pandas as pd

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename='c:/Users/group/Downloads/experiment_results.log'
    )

def main():
    setup_logging()
    os.makedirs('c:/Users/group/Downloads/images', exist_ok=True)
    os.makedirs('c:/Users/group/Downloads/results', exist_ok=True)
    
    runner = ExperimentRunner()
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
        
        # Save results
        for name, df in results.items():
            df.to_csv(f'c:/Users/group/Downloads/results/{name}_results.csv')
        
        logging.info("All experiments completed successfully")
        
    except Exception as e:
        logging.error(f"Error during experiments: {str(e)}")
        raise

if __name__ == "__main__":
    main()