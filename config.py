import os
import pathlib

# Get the home directory and create the base path
HOME = str(pathlib.Path.home())
BASE_DIR = os.path.join(HOME, 'APP')

# Define all directory paths
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
IMAGES_DIR = os.path.join(BASE_DIR, 'images')

# Define log file path
LOG_FILE = os.path.join(BASE_DIR, 'experiment_results.log')

# Create directories if they don't exist
for directory in [RESULTS_DIR, IMAGES_DIR]:
    os.makedirs(directory, exist_ok=True)