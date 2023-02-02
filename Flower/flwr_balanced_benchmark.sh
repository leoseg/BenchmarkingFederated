#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:../."
DATA_PATH=$1
NUM_CLIENTS=$2
NUM_ROUNDS=$3
source activate ../venvBM
python ../scripts/partion_data.py --num_clients $NUM_CLIENTS  --data_path $DATA_PATH
echo "Benchmark model metrics"
for repeat in range{0..1}
do
  python server.py --data_path $DATA_PATH --random_state $repeat --num_clients $NUM_CLIENTS --num_rounds $NUM_ROUNDS --system_metrics false &
  sleep 3
  for ((i=1;i<=$NUM_CLIENTS;i++))
  do
    python client.py --client_index $i --data_path $DATA_PATH --run_repeat $repeat --system_metrics false &
  done
done
echo "Benchmark system metrics"
for repeat in range{0..100}
do
  python server.py --data_path $DATA_PATH --run_repeat $repeat --num_clients 1 --num_rounds $NUM_ROUNDS --system_metrics true &
  sleep 3
  python client.py --client_index 0 --data_path $DATA_PATH --random_state $repeat --system_metrics true &
done