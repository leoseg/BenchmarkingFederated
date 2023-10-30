#!/bin/bash
#SBATCH -J tff_dp
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
cd Flower || exit
clients_config=(3)
rounds_config=(1 2 5 10)
#if [ "$USECASE" -eq "4" ]
#then
#    noises=(2.5 3.5 4.5 5.5 6.0)
#elif [ "$USECASE" -eq "3" ]
#then
#    noises=(1.5 2.0 2.5 3.0 3.5)
#elif [ "$USECASE" -eq "2" ]
#then
#    noises=(1.5 2.5 3.5 4.0 5.0)
#else
#    noises=(2.0 3.0 4.0 5.0 6.0)
#fi
noises=(0.5 1.0 2.5 3.5 5.0)
# Loops trough round and number of clients configuration
for num_rounds in "${rounds_config[@]}";
do
  for noise in "${noises[@]}";
  do
    bash flwr_balanced_benchmark.sh $DATA_PATH 3 $num_rounds $WANDB_API_KEY $NUM_REPEATS $SYSTEM_ONLY $noise
  done
done
