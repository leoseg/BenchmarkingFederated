#!/bin/bash
#SBATCH -J tff_bal
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --partition=clara
#SBATCH --cpus-per-task=11
#SBATCH --mem=100G
module load Python/3.10.4-GCCcore-11.3.0
export PYTHONPATH="${PYTHONPATH}:../."
WANDB_API_KEY=$1
NUM_REPEATS=$2
USECASE=$3
export USECASE=$USECASE
DATA_PATH=$4
SYSTEM_ONLY=$5
cd ..
python3.10 -m venv venvtff
source venvtff/bin/activate
python3 -c 'import sys; print(sys.version_info[:])'
pip3 install --upgrade pip
pip install -e utils
pip3 install -r requirements.txt
cd TensorflowFederated || exit
#if [ $3 =  "1" ] ||  [ $3 = "3" ] || [ $3 = "4" ]; then
#   round_config=(1)
#elif [  $3 = "2" ]; then
#   round_config=(1 4)
#fi
clients_config=(3 5 10 50)
noises=(0.0 0.25 0.5 0.75 1.0 1.25)
# Loops trough round and number of clients configuration
for clients in "${clients_config[@]}";
do
  for noise in "${noises[@]}";
  do
    bash tff_dp.sh $DATA_PATH $noise $clients $WANDB_API_KEY $NUM_REPEATS $SYSTEM_ONLY
  done
done

