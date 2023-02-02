#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:../."
DATA_PATH=$1
NUM_CLIENTS=$2
NUM_ROUNDS=$3
source activate ../venvBM
python ../scripts/partion_data.py --num_clients $NUM_CLIENTS --data_path $DATA_PATH
echo "Benchmark model metrics"
for repeat in range{0..1}
do
  for ((i=1;i<=$NUM_CLIENTS;i++))
  do
    port=$((8000 + $i))
    python worker_serivce.py --port $port --num_rounds $NUM_ROUNDS --client_index $i --data_path $DATA_PATH --random_state $repeat &
  done
  python train_gen_expr.py --num_clients $NUM_CLIENTS --num_rounds $NUM_ROUNDS --data_path $DATA_PATH --run_repeat $repeat --system_metrics false
  pkill worker_serivce
done
echo "Benchmark system metrics"
for repeat in range{0..10}
do
  python worker_serivce.py --port 8001 --num_rounds $NUM_ROUNDS --client_index 0 --data_path $DATA_PATH --random_state $repeat &
  python train_gen_expr.py --num_clients 1 --num_rounds $NUM_ROUNDS --data_path $DATA_PATH --run_repeat $repeat --system_metrics true
  pkill worker_serivce
done