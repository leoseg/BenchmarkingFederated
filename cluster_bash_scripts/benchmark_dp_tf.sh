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
cd TensorflowFederated || exit
#clients_config=(5)
if [ $3 =  "1" ] ||  [ $3 = "3" ] || [ $3 = "4" ]; then
   rounds_config=(1 2 5 10)
elif [  $3 = "2" ]; then
   rounds_config=(1 2 4 8)
fi
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
noises=(0.01 0.03 0.05 0.07 0.085  0.1)
# Loops trough round and number of clients configuration
for num_rounds in "${rounds_config[@]}";
do
  for noise in "${noises[@]}";
  do
    bash tff_dp.sh $DATA_PATH $noise 5 $WANDB_API_KEY $NUM_REPEATS $SYSTEM_ONLY $num_rounds
  done
done
