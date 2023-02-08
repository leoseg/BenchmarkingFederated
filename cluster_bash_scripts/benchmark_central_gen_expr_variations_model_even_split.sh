#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=training_unsupervised
#SBATCH --partition=clara
#SBATCH --mem=50G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=leoseeger16@gmail.com
module load Python/3.10.4-GCCcore-11.3.0
export PYTHONPATH="${PYTHONPATH}:../."
cd ..
python3.10 -m venv venv
source venv/bin/activate
pip3 install --upgrade pip
pip install -e utils
pip3 install -r requirements.txt
WANDB_API_KEY=$1
cd CentralizedApproach || exit
python train_model_wandb_gen_expr_take_data_from_all.py --num_nodes 512 --dropout_rate 0.3 --l1_v 0.0
python train_model_wandb_gen_expr_take_data_from_all.py --num_nodes 1024 --dropout_rate 0.3 --l1_v 0.0
python train_model_wandb_gen_expr_take_data_from_all.py --num_nodes 256 --dropout_rate 0.5 --l1_v 0.005
