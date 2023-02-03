#!/bin/bash
set -e
export PYTHONPATH="${PYTHONPATH}:../."
DATA_PATH=$1
NUM_CLIENTS=$2
NUM_ROUNDS=$3
REPEATS=$5
DATA_NAME=$(basename "$DATA_PATH")
echo "Starting flwr experiment with num clients ${NUM_CLIENTS} num rounds ${NUM_ROUNDS} and data ${DATA_NAME} and ${REPEATS} repeats"
source activate ../venvBM
python ../scripts/partition_data.py --num_clients $NUM_CLIENTS  --data_path $DATA_PATH
echo "Benchmark model metrics"
for (( repeat = 0; repeat < $REPEATS; repeat++ ))
do
  echo "Start repeat ${repeat}"
  echo "Start server for repeat ${repeat}"
  python server.py --data_path $DATA_PATH --run_repeat $repeat --num_clients $NUM_CLIENTS --num_rounds $NUM_ROUNDS &
  sleep 3
  for ((i=1;i<=$NUM_CLIENTS;i++))
  do
    echo "Start client ${i}"
    client_index=$(($i -1))
    python client.py --client_index $client_index --data_path $DATA_PATH --run_repeat $repeat &
  done
  wait
  echo "Repeat ${repeat} complete"
done
echo "Benchmark system metrics"
for (( repeat = 0; repeat < $REPEATS; repeat++ ))
do
  echo "Start repeat ${repeat}"
  echo "Creating server"
  python server.py --data_path $DATA_PATH --run_repeat $repeat --num_clients 1 --num_rounds $NUM_ROUNDS --system_metrics true &
  server_id=$!
  sleep 3
  echo "Start client"
  python client.py --client_index 0 --data_path $DATA_PATH --random_state $repeat --system_metrics true &
  client_id=$!
  psrecord $client_id --log ",timelogs/flwr_client_${DATA_NAME}_${NUM_CLIENTS}_${NUM_ROUNDS}_repeat_${repeat}.txt" &
  psrecord $server_id --log "timelogs/flwr_server_${DATA_NAME}_${NUM_CLIENTS}_${NUM_ROUNDS}_repeat_${repeat}.txt" &
done