#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=training_unsupervised
#SBATCH --partition=clara
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=11
#SBATCH --mem=50G
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=leoseeger16@gmail.com
module load Python/3.10.4-GCCcore-11.3.0
export PYTHONPATH="${PYTHONPATH}:../."
WANDB_API_KEY=$1
NUM_REPEATS=$2
echo $WANDB_API_KEY
cd ..
python3.10 -m venv venv
source venvBM/bin/activate
python3 -c 'import sys; print(sys.version_info[:])'
pip3 install --upgrade pip
pip install -e utils
pip3 install -r requirements.txt
cd Flower || exit=
for rounds in {1,2,5,10}
do
  for client_num in {3,5,10}
  do
    bash flwr_balanced_benchmark.sh "../DataGenExpression/Alldata.csv" $client_num $rounds $WANDB_API_KEY $NUM_REPEATS
  done
done
