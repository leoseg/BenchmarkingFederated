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
USECASE=$3
export USECASE=$USECASE
DATA_PATH=$4
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
for rounds in {1,3,10}
do
  for unweight_step in {0,2,4,6,8,9,10}
  do
    bash tff_unbalanced_benchmark.sh $DATA_PATH 2 $rounds $WANDB_API_KEY $NUM_REPEATS $unweight_step
  done
done

