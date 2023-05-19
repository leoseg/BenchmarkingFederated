#!/bin/bash
#SBATCH -J tff_bal
#SBATCH --time=2-00:00:00
#SBATCH --partition=clara
#SBATCH --cpus-per-task=60
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
# Choose rounds configuration depending on usecase
if [ $3 =  "1" ] ||  [ $3 = "3" ] || [ $3 = "4" ]; then
   round_config=(1 2 5 10)
elif [  $3 = "2" ]; then
   round_config=(1 2 4 8)
fi
# Loops trough round and number of clients configuration
for rounds in "${round_config[@]}";
do
  for client_num in {3,5,10, 50}
  do
    bash tff_balanced_benchmark.sh $DATA_PATH $client_num $rounds $WANDB_API_KEY $NUM_REPEATS $SYSTEM_ONLY
  done
done

