#!/bin/bash
set -e
export PYTHONPATH="${PYTHONPATH}:../."
DATA_PATH=$1
NUM_CLIENTS=$2
NUM_ROUNDS=$3
WANDB_API_KEY=$4
REPEATS=$5
DATA_NAME=$(basename "$DATA_PATH")
echo "Starting tff experiment with num clients ${NUM_CLIENTS} num rounds ${NUM_ROUNDS} and data ${DATA_NAME} and ${REPEATS} repeats"
source activate ../venvBM
python ../scripts/partition_data.py --num_clients $NUM_CLIENTS --data_path $DATA_PATH
#echo "Benchmark model metrics"
#for (( repeat = 0; repeat < $REPEATS; repeat++ ))
#do
#  echo "Start repeat ${repeat}"
#  for ((i=1;i<=$NUM_CLIENTS;i++))
#  do
#    port=$((8000 + $i))
#    echo "Creating worker ${i} with port ${port}"
#    client_index=$(($i -1))
#    python worker_service.py --port $port --num_rounds $NUM_ROUNDS --client_index $client_index --data_path $DATA_PATH --random_state $repeat &
#  done
#  sleep 6
#  echo "Start training for repeat ${repeat}"
#  python tff_benchmark_gen_express.py --num_clients $NUM_CLIENTS --num_rounds $NUM_ROUNDS --data_path $DATA_PATH --run_repeat $repeat
#  pkill worker_service
#  echo "Repeat ${repeat} complete"
#done
echo "Benchmark system metrics"
for (( repeat = 0; repeat < $REPEATS; repeat++ ))
do
  echo "Start repeat ${repeat}"
  echo "Creating single worker service"
  python worker_service.py --port 8001 --num_rounds $NUM_ROUNDS --client_index 0 --data_path $DATA_PATH --random_state $repeat &
  worker_id=$!
  echo "Start training"
  python tff_benchmark_gen_express.py --num_clients 1 --num_rounds $NUM_ROUNDS --data_path $DATA_PATH --run_repeat $repeat --system_metrics true &
  train_id=$!
  psrecord $worker_id --log "timelogs/tff_worker_${DATA_NAME}_${NUM_CLIENTS}_${NUM_ROUNDS}_repeat_${repeat}.txt" &
  psrecord $train_id --log "timelogs/tff_train_${DATA_NAME}_${NUM_CLIENTS}_${NUM_ROUNDS}_repeat_${repeat}.txt" &
  wait $train_id
  pkill worker_service
done