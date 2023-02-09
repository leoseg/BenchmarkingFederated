#!/bin/bash
#SBATCH --job-name=tff_benchmark
#SBATCH --partition=clara
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=11
#SBATCH --mem-per-cpu=7G
#SBATCH --ntasks=12
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=leoseeger16@gmail.com
module load Python/3.10.4-GCCcore-11.3.0
export PYTHONPATH="${PYTHONPATH}:../."
WANDB_API_KEY=$1
NUM_REPEATS=$2
echo $WANDB_API_KEY
chmod 777 benchmark_tff_gen_expr.sh
cd ..
python3.10 -m venv venvtff
source venvtff/bin/activate
python3 -c 'import sys; print(sys.version_info[:])'
pip3 install --upgrade pip
pip install -e utils
pip3 install -r requirements.txt
cd TensorflowFederated || exit
for rounds in {1,2,5,10}
do
  for client_num in {3,5,10}
  do
    echo "Starten srun with ${client_num} clients and ${rounds} rounds"
    srun --cpus-per-task=$SLURM_CPUS_PER_TASK --ntasks=1 tff_balanced_benchmark.sh "../DataGenExpression/Alldata.csv" $client_num $rounds $WANDB_API_KEY $NUM_REPEATS &
  done
done
wait
