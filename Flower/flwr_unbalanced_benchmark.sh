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
echo "Usecase ${USECASE}"
echo "Starting unbalanced flwr experiment with num clients ${NUM_CLIENTS} num rounds ${NUM_ROUNDS} and data ${DATA_NAME} and ${REPEATS} repeats"
# Creates partitions and saves the row indices of each partition to file so it can be read from clients
python ../scripts/partition_data.py --num_clients $NUM_CLIENTS  --data_path $DATA_PATH --unweighted_step $UNWEIGHTED_STEP
echo "Benchmark model metrics"
for (( repeat = 0; repeat < $REPEATS; repeat++ ))
do
  echo "Start server for repeat model metrics ${repeat} num clients ${NUM_CLIENTS} num rounds ${NUM_ROUNDS} and data ${DATA_NAME} and unweighted step ${UNWEIGHTED_STEP}"
  python server.py --data_path $DATA_PATH --run_repeat $repeat --num_clients $NUM_CLIENTS --num_rounds $NUM_ROUNDS --unweighted_percentage $UNWEIGHTED_STEP &
  sleep 300
  echo "Start repeat ${repeat}"
  for ((i=1;i<=$NUM_CLIENTS;i++))
  do
    echo "Start client ${i}"
    client_index=$(($i -1))
    python client.py --client_index $client_index --data_path $DATA_PATH --run_repeat $repeat --unweighted true &
  done
  wait
  echo "Repeat model metrics ${repeat} num clients ${NUM_CLIENTS} num rounds ${NUM_ROUNDS} and data ${DATA_NAME} complete"
  echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
done
echo "---------------------------------------------------------------------------------------------------------"
sleep 20