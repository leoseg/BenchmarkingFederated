#!/bin/bash
#SBATCH -J tff
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=11
#SBATCH --mem=50G
module load Python/3.10.4-GCCcore-11.3.0
export PYTHONPATH="${PYTHONPATH}:../."
WANDB_API_KEY=$1
NUM_REPEATS=$2
echo $WANDB_API_KEY
cd ..
python3.10 -m venv venvtff
source venvtff/bin/activate
python3 -c 'import sys; print(sys.version_info[:])'
pip3 install --upgrade pip
pip install -e utils
pip3 install -r requirements.txt
cd TensorflowFederated || exit
rounds=10
for client_num in {3,5,10}
do
  echo "Start run with ${client_num} clients and ${rounds} rounds"
  bash "../DataGenExpression/Alldata.csv" $client_num $rounds $WANDB_API_KEY $NUM_REPEATS
done
bash "../DataGenExpression/Alldata.csv" 10 5 $WANDB_API_KEY $NUM_REPEATS

