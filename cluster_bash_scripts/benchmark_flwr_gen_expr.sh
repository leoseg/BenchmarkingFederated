#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=training_unsupervised
#SBATCH --partition=clara
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=leoseeger16@gmail.com
module load Python/3.10.4-GCCcore-11.3.0
python3 -m venv venv
export PYTHONPATH="${PYTHONPATH}:../."
source activate venv
pip3 install --upgrade pip
pip3 install -r ../requirements.txt
WANDB_API_KEY=$WANDB_API_KEY
cd ..
cd Flower || exit
for client_num in {3..10}
do
  for rounds in {1,2,5,10}
  do
    bash flwr_balanced_benchmark.sh "../DataGenExpression/Alldata.csv" $client_num $rounds $WANDB_API_KEY 100
  done
done
