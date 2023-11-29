#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=central_syste,
#SBATCH --cpus-per-task=1
#SBATCH --partition=clara
#SBATCH --time=2-00:00:00
#SBATCH --mem-per-cpu=50G
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=leoseeger16@gmail.com
module load Python/3.10.4-GCCcore-11.3.0
export PYTHONPATH="${PYTHONPATH}:../."
cd ..
# Create environment and install packages
python3.10 -m venv venv
source venvBM/bin/activate
pip3 install --upgrade pip
pip install -e utils
pip3 install -r requirements.txt
WANDB_API_KEY=$1
REPEATS=$2
USECASE=$3
export USECASE=$USECASE
cd CentralizedApproach || exit
for (( repeat = 0; repeat < $REPEATS; repeat++ ))
do
  python benchmark_central_system_metrics.py --run_repeat $repeat &
  process_id=$!
  # Gets string of available cpus
  read cpu_array <<< $(taskset -pc $process_id | awk '{print $NF}')
  # Reads this to an array
  cpu_num=$(echo $cpu_array | cut -d'-' -f1)
  # Bind the process to one cpu
  taskset -c -pa $cpu_num $process_id
  # Start recording memory stats
  psrecord $process_id --log "timelogs/central_model_repeat_${repeat}.txt" --interval 0.5
  project_name="benchmark-central_system_metrics"
  group_name="USECASE_${USECASE}"
  # Reads memory stats from file written by psrecord
  python ../scripts/mem_data_to_wandb.py --logs_path "timelogs/central_model_repeat_${repeat}.txt" --project_name $project_name --run_name "run_${repeat}" --group_name $group_name --memory_type "central"
done
