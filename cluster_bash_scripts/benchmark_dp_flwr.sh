#!/bin/bash
#SBATCH -J tff_dp
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --partition=clara
#SBATCH --cpus-per-task=11
#SBATCH --mem=100G
#SBATCH --mail-user=leoseeger16@googlemail.com
#SBATCH --mail-type=FAIL
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
if [ $3 =  "1" ] ||  [ $3 = "3" ] || [ $3 = "4" ]; then
   rounds_config=(1 2 5 10)
elif [  $3 = "2" ]; then
   rounds_config=(4 8)
fi
noises=(0.01 0.03 0.05 0.07 0.085  0.1)
# Loops trough round and number of clients configuration
for num_rounds in "${rounds_config[@]}";
do
  for noise in "${noises[@]}";
  do
    bash flwr_balanced_benchmark.sh $DATA_PATH 5 $num_rounds $WANDB_API_KEY $NUM_REPEATS $SYSTEM_ONLY $noise
  done
done
