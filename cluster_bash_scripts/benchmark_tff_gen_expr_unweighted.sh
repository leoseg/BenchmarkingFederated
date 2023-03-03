#!/bin/bash
#SBATCH -J tff_inb
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
cd ..
# Create environment and install packages
python3.10 -m venv venvtff
source venvtff/bin/activate
python3 -c 'import sys; print(sys.version_info[:])'
pip3 install --upgrade pip
pip install -e utils
pip3 install -r requirements.txt
cd TensorflowFederated || exit
# Choose rounds configuration depending on usecase
if [ $3 =  "1" ] ||  [ $3 = "3" ]; then
   round_config=(1 2 5 10)
elif [  $3 = "2" ]; then
   round_config=(1 2 4 8)
fi
# Choose step config for "unweighting" the class distribution on each client
if [ $3 =  "1" ] ||  [ $3 = "2" ]; then
   unweight_config=(0 2 4 6 8 9 10)
   num_clients=2
elif [  $3 = "3" ]; then
   unweight_config=(0 1 2 3 4)
   num_clients=5
fi
# Loops trough configurations
for rounds in "${round_config[@]}";
do
  for unweight_step in "${unweight_config[@]}";
  do
    bash tff_unbalanced_benchmark.sh $DATA_PATH $num_clients $rounds $WANDB_API_KEY $NUM_REPEATS $unweight_step
  done
done

