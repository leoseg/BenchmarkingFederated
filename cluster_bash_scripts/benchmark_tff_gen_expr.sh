#!/bin/bash
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
    sbatch tff_balanced_benchmark.sh "../DataGenExpression/Alldata.csv" $client_num $rounds $WANDB_API_KEY $NUM_REPEATS &
  done
done
wait
