#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=training_unsupervised
#SBATCH --partition=clara
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=leoseeger16@gmail.com
#module load Anaconda3
export PYTHONPATH="${PYTHONPATH}:../."
source activate venv
pip3 install -r ../requirements.txt
WANDB_API_KEY=$WANDB_API_KEY
python ../CentralizedApproach/benchmark_central.py --num_nodes 512 --dropout_rate 0.3 --l1_v 0.005 --data_path "../DataGenExpression/Alldata.csv"
python ../CentralizedApproach/benchmark_central.py --num_nodes 1024 --dropout_rate 0.3 --l1_v 0.0 --data_path "../DataGenExpression/Alldata.csv"
python ../CentralizedApproach/benchmark_central.py --num_nodes 512 --dropout_rate 0.5 --l1_v 0.005 --data_path "../DataGenExpression/Alldata.csv"

