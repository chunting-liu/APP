# Aggregate Production Planning with Nonlinear Carbon Emission Functions

This project implements an enhanced Aggregate Production Planning (APP) model that integrates multiple carbon emission functions to optimize production planning while considering environmental sustainability. The model provides a robust and flexible tool for balancing economic objectives with environmental considerations.

## Overview

The project addresses the limitations of conventional APP models by incorporating diverse carbon emission patterns:
- Linear emission functions
- Quadratic emission functions
- Exponential emission functions
- Logarithmic emission functions
- Piecewise linear emission functions

This approach enables more accurate modeling of real-world emission behaviors across different industries and production processes.

## Features

- **Multiple Emission Patterns**: Supports various emission functions to accurately model different industrial processes
- **Stochastic Demand Handling**: Incorporates demand uncertainty through scenario-based optimization
- **Cost Optimization**: Minimizes total costs including:
  - Production costs
  - Inventory holding costs
  - Backordering costs
  - Energy consumption costs
  - Carbon emission costs
  - Production adjustment costs
- **Constraint Management**: Handles multiple operational constraints including:
  - Production capacity limits
  - Backorder limits
  - Carbon emission caps

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running Experiments

To run the experiments and generate results:

```bash
python run_experiments.py
```

This will:
- Set up logging
- Create necessary directories for results and images
- Run various analyses including:
  - Emission pattern analysis
  - Industry case studies
  - Sustainability trade-offs
  - Demand uncertainty effects
- Save results to CSV files in the results directory

## Project Structure

- `app_model.py`: Core implementation of the APP model
- `config.py`: Configuration settings and parameters
- `emission_functions.py`: Implementation of different emission functions
- `experiments.py`: Experimental setup and analysis functions
- `generate_results.py`: Results generation and processing
- `visualization.py`: Visualization utilities
- `run_experiments.py`: Main script to run experiments

## Model Details

The model optimizes production planning by:
1. Integrating multiple carbon emission functions
2. Handling stochastic demand scenarios
3. Balancing economic and environmental objectives
4. Considering operational constraints
5. Supporting industry-specific emission patterns

## Results

Results are saved in the following formats:
- `emission_patterns_results.csv`: Analysis of different emission patterns
- `industry_cases_results.csv`: Industry-specific case studies
- `sustainability_results.csv`: Sustainability trade-off analysis
- `uncertainty_results.csv`: Effects of demand uncertainty

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```
Liu, C., & Minner, S. (2024). Aggregate Production Planning Considering Different Carbon Emission Patterns.
```
