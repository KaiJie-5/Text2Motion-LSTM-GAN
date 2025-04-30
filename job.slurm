#!/bin/bash

#SBATCH --job-name=my_job       # Job name
#SBATCH --output=Case_19_output.log      # Output log file
#SBATCH --error=Case_19_error.log        # Error log file
#SBATCH --time=2-02:00:00          # Maximum run time (hh:mm:ss)
#SBATCH --partition=gpu    # Partition name
#SBATCH --nodes=1                # Number of nodes
#SBATCH --gres=gpu:1             # Number of GPUs (if needed)

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=kjl1a21@soton.ac.uk

# Load necessary modules
module load cuda/12.2 

# Initialize Conda and activate environment
eval "$(conda shell.bash hook)"
conda activate myenv

# Run your command
python execute_model.py