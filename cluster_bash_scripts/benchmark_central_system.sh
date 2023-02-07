#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=training_unsupervised
#SBATCH --partition=clara
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=leoseeger16@gmail.com
module load Python/3.10.4-GCCcore-11.3.0
export PYTHONPATH="${PYTHONPATH}:../."
cd ..
python3 -m venv venv
python3.10 -m venv venv
source venv/bin/activate
pip3 install --upgrade pip
pip install -e utils
pip3 install -r requirements.txt
DATA_PATH="../DataGenExpression/Alldata.csv"
WANDB_API_KEY=$1
REPEATS=$2
DATA_NAME=$(basename "$DATA_PATH" .csv)
cd CentralizedApproach || exit
for (( repeat = 0; repeat < $REPEATS; repeat++ ))
do
  python benchmark_central_system_metrics.py --run_repeat $repeat --data_path $DATA_PATH &
  psrecord $! --log "timelogs/central_model_repeat_${repeat}.txt" --interval 0.5
  project_name="benchmark-central_${DATA_NAME}_system_metrics"
  python ../scripts/mem_data_to_wandb.py --logs_path "timelogs/central_model_repeat_${repeat}.txt" --project_name $project_name --run_name "run_${repeat}" --group_name "central"  --memory_type "central"
done