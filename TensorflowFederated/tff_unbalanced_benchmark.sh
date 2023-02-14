#!/bin/bash
set -e
export PYTHONPATH="${PYTHONPATH}:../."
export TF_CPP_MIN_LOG_LEVEL=3
DATA_PATH=$1
NUM_CLIENTS=$2
NUM_ROUNDS=$3
WANDB_API_KEY=$4
REPEATS=$5
UNWEIGHTED_STEP=$6
DATA_NAME=$(basename "$DATA_PATH" .csv)
echo "Starting unbalanced tff experiment with num clients ${NUM_CLIENTS} num rounds ${NUM_ROUNDS} and data ${DATA_NAME} and ${REPEATS} repeats"
python ../scripts/partition_data.py --num_clients $NUM_CLIENTS --data_path $DATA_PATH --unweighted_step $UNWEIGHTED_STEP --label_name "Condition"
echo "Benchmark model metrics"
for (( repeat = 0; repeat < $REPEATS; repeat++ ))
do
  echo "Start repeat model metrics ${repeat} num clients ${NUM_CLIENTS} num rounds ${NUM_ROUNDS} and data ${DATA_NAME} and unweighted step ${UNWEIGHTED_STEP}"
  for ((i=1;i<=$NUM_CLIENTS;i++))
  do
    port=$((8000 + $i))
    echo "Creating worker ${i} with port ${port}"
    client_index=$(($i -1))
    python worker_service.py --port $port --num_rounds $NUM_ROUNDS --client_index $client_index --data_path $DATA_PATH --random_state $repeat &
  done
  sleep 6
  echo "Start training for repeat ${repeat}"
  python tff_benchmark_gen_express.py --num_clients $NUM_CLIENTS --num_rounds $NUM_ROUNDS --data_path $DATA_PATH --run_repeat $repeat --unweighted_percentage $UNWEIGHTED_STEP
  pkill worker_service
  echo "Repeat model metrics ${repeat} num clients ${NUM_CLIENTS} num rounds ${NUM_ROUNDS} and data ${DATA_NAME} complete"
  echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
done
echo "---------------------------------------------------------------------------------------------------------"
