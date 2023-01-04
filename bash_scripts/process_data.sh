#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=process_data
#SBATCH --partition=clara
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=leoseeger16@gmail.com
module load Anaconda3
source activate venv
pip3 install -r requirements.txt
python .././process_data_script.py
