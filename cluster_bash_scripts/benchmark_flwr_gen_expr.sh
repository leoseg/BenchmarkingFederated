#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=flwr_bal
#SBATCH --partition=clara
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=11
#SBATCH --mem=50G
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=leoseeger16@gmail.com
module load Python/3.10.4-GCCcore-11.3.0
export PYTHONPATH="${PYTHONPATH}:../."
WANDB_API_KEY=$1
NUM_REPEATS=$2
USECASE=$3
export USECASE=$USECASE
DATA_PATH=$4
echo $WANDB_API_KEY
cd ..
python3.10 -m venv venvFlwr
source venvFlwr/bin/activate
python3 -c 'import sys; print(sys.version_info[:])'
pip3 install --upgrade pip
pip install -e utils
pip3 install -r requirements.txt
cd Flower || exit
if [ $3 =  "1" ] ||  [ $3 = "3" ]; then
   round_config=(1 2 5 10)
elif [  $3 = "2" ]; then
   round_config=(1 2 4 8)
fi
for rounds in "${round_config[@]}";
do
  for client_num in {3,5,10}
  do
    bash flwr_balanced_benchmark.sh $DATA_PATH $client_num $rounds $WANDB_API_KEY $NUM_REPEATS
  done
done
