#!/bin/bash
#SBATCH --job-name=tff_benchmark
#SBATCH --partition=clara
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=11
#SBATCH --mem-per-cpu=4g
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=leoseeger16@gmail.com
set -e
export PYTHONPATH="${PYTHONPATH}:../."
export TF_CPP_MIN_LOG_LEVEL=3
DATA_PATH=$1
NUM_CLIENTS=$2
NUM_ROUNDS=$3
WANDB_API_KEY=$4
REPEATS=$5
DATA_NAME=$(basename "$DATA_PATH" .csv)
echo "Starting tff experiment with num clients ${NUM_CLIENTS} num rounds ${NUM_ROUNDS} and data ${DATA_NAME} and ${REPEATS} repeats"
python ../scripts/partition_data.py --num_clients $NUM_CLIENTS --data_path $DATA_PATH
echo "Benchmark model metrics"
for (( repeat = 0; repeat < $REPEATS; repeat++ ))
do
  echo "Start repeat model metrics ${repeat} num clients ${NUM_CLIENTS} num rounds ${NUM_ROUNDS} and data ${DATA_NAME}"
  for ((i=1;i<=$NUM_CLIENTS;i++))
  do
    port=$((8000 + $i))
    echo "Creating worker ${i} with port ${port}"
    client_index=$(($i -1))
    python worker_service.py --port $port --num_rounds $NUM_ROUNDS --client_index $client_index --data_path $DATA_PATH --random_state $repeat &
  done
  sleep 6
  echo "Start training for repeat ${repeat}"
  python tff_benchmark_gen_express.py --num_clients $NUM_CLIENTS --num_rounds $NUM_ROUNDS --data_path $DATA_PATH --run_repeat $repeat
  pkill worker_service
  echo "Repeat model metrics ${repeat} num clients ${NUM_CLIENTS} num rounds ${NUM_ROUNDS} and data ${DATA_NAME} complete"
  echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
done
echo "---------------------------------------------------------------------------------------------------------"
echo "Benchmark system metrics"
for (( repeat = 0; repeat < $REPEATS; repeat++ ))
do
  echo "Start repeat system metrics ${repeat} num clients ${NUM_CLIENTS} num rounds ${NUM_ROUNDS} and data ${DATA_NAME}"
  rm timelogs/tff_logs_time.txt
  echo "Creating single worker service"
  python worker_service.py --port 8001 --num_rounds $NUM_ROUNDS --client_index 1 --data_path $DATA_PATH --random_state $repeat &
  worker_id=$!
  echo "Start training"
  python tff_benchmark_gen_express.py --num_clients $NUM_CLIENTS --num_rounds $NUM_ROUNDS --data_path $DATA_PATH --run_repeat $repeat --system_metrics true &
  train_id=$!
  worker_time_logs="timelogs/tff_worker_${DATA_NAME}_${NUM_CLIENTS}_${NUM_ROUNDS}_repeat_${repeat}.txt"
  train_time_logs="timelogs/tff_train_${DATA_NAME}_${NUM_CLIENTS}_${NUM_ROUNDS}_repeat_${repeat}.txt"
  psrecord $worker_id --log $worker_time_logs --interval 0.5 &
  psrecord $train_id --log $train_time_logs --interval 0.5
  pkill worker_service
  project_name="benchmark_rounds_${NUM_ROUNDS}_${DATA_NAME}_system_metrics"
  run_name="run_${repeat}"
  python ../scripts/mem_data_to_wandb.py --logs_path $worker_time_logs --project_name $project_name --run_name $run_name --group_name "tff_${NUM_CLIENTS}"  --memory_type "client"
  python ../scripts/mem_data_to_wandb.py --logs_path $train_time_logs --project_name $project_name  --run_name $run_name --group_name "tff_${NUM_CLIENTS}"  --memory_type "server"
  echo "Repeat system metrics ${repeat} num clients ${NUM_CLIENTS} num rounds ${NUM_ROUNDS} and data ${DATA_NAME} complete"
  echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
done