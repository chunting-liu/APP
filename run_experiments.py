from experiments import ExperimentRunner
import logging
import os
import pandas as pd
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
            df.to_csv(os.path.join(RESULTS_DIR, f'{name}_results.csv'))
        
        logging.info("All experiments completed successfully")
        
    except Exception as e:
        logging.error(f"Error during experiments: {str(e)}")
        raise

if __name__ == "__main__":
    main()