#!/bin/bash
set -e
export PYTHONPATH="${PYTHONPATH}:../."
export TF_CPP_MIN_LOG_LEVEL=3
DATA_PATH=$1
NUM_CLIENTS=$2
NUM_ROUNDS=$3
WANDB_API_KEY=$4
REPEATS=$5
SYSTEM_ONLY=$6
DATA_NAME=$(basename "$DATA_PATH" .csv)
echo "Starting flwr experiment with num clients ${NUM_CLIENTS} num rounds ${NUM_ROUNDS} and data ${DATA_NAME} and ${REPEATS} repeats"
# Creates partitions and saves the row indices of each partition to file so it can be read from clients
python ../scripts/partition_data.py --num_clients $NUM_CLIENTS  --data_path $DATA_PATH
# Benchmark model performance metrics if system only is not set
if [ $SYSTEM_ONLY != "1" ]; then
  echo "Benchmark model metrics"
  for (( repeat = 0; repeat < $REPEATS; repeat++ ))
  do
    echo "Start server for repeat model metrics ${repeat} num clients ${NUM_CLIENTS} num rounds ${NUM_ROUNDS} and data ${DATA_NAME}"
    python server.py --data_path $DATA_PATH --run_repeat $repeat --num_clients $NUM_CLIENTS --num_rounds $NUM_ROUNDS &
    sleep  $(($NUM_CLIENTS * 2))
    echo "Start repeat ${repeat}"
    for ((i=1;i<=$NUM_CLIENTS;i++))
    do
      echo "Start client ${i}"
      client_index=$(($i -1))
      python client.py --client_index $client_index --data_path $DATA_PATH --run_repeat $repeat &
    done
    wait
    echo "Repeat model metrics ${repeat} num clients ${NUM_CLIENTS} num rounds ${NUM_ROUNDS} and data ${DATA_NAME} complete"
    echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  done
  echo "---------------------------------------------------------------------------------------------------------"
fi
if [ $SYSTEM_ONLY != "2" ]; then
  echo "Benchmark system metrics"
  for (( repeat = 0; repeat < $REPEATS; repeat++ ))
  do
    echo "Start repeat system metrics ${repeat} num clients ${NUM_CLIENTS} num rounds ${NUM_ROUNDS} and data ${DATA_NAME}"
    rm -f timelogs/flw_logs_time.txt
    echo "Creating server"
    python server.py --data_path $DATA_PATH --run_repeat $repeat --num_clients $NUM_CLIENTS --num_rounds $NUM_ROUNDS --system_metrics true &
    server_id=$!
    # Reads in all cpu available as string
    read cpu_available <<< $(taskset -pc $server_id | awk '{print $NF}')
    # Converts that string into an array
    cpu_arr=(${cpu_available//,/ })
    # Reads in two CPU numbers for latter assignment, one for the server one for the client, because sometimes
    # the taskset command returns a range of CPU's in form of '5-7' the cut command is needed for extracting only one number
    cpu_num_1=$(echo ${cpu_arr[0]} | cut -d'-' -f1)
    cpu_num_2=$(echo ${cpu_arr[1]} | cut -d'-' -f1)
    # Bounds the CPU with 'cpu_num_1' to the server process
    taskset -c -pa $cpu_num_1 $server_id
    #sleep 3
    echo "Start client"
    python client.py --client_index 1 --data_path $DATA_PATH --run_repeat $repeat --system_metrics true &
    client_id=$!
    # Bind client process to cpu num 2
    taskset -c -pa $cpu_num_2 $client_id
    client_time_logs="timelogs/flwr_client_${DATA_NAME}_${NUM_CLIENTS}_${NUM_ROUNDS}_repeat_${repeat}.txt"
    server_time_logs="timelogs/flwr_server_${DATA_NAME}_${NUM_CLIENTS}_${NUM_ROUNDS}_repeat_${repeat}.txt"
    # Record memory from client and server process and log to file
    psrecord $client_id --log $client_time_logs  --interval 0.5 &
    psrecord $server_id --log $server_time_logs --interval 0.5 &
    wait
    project_name="usecase_${USECASE}_benchmark_rounds_${NUM_ROUNDS}_${DATA_NAME}_system_metrics"
    run_name="run_${repeat}"
    # Read files logged from psutil to wandb
    python ../scripts/mem_data_to_wandb.py --logs_path $client_time_logs --project_name $project_name --run_name $run_name --group_name "flwr_${NUM_CLIENTS}"  --memory_type "client"
    python ../scripts/mem_data_to_wandb.py --logs_path $server_time_logs --project_name $project_name  --run_name $run_name --group_name "flwr_${NUM_CLIENTS}"  --memory_type "server"
    echo "Repeat system metrics ${repeat} num clients ${NUM_CLIENTS} num rounds ${NUM_ROUNDS} and data ${DATA_NAME} complete"
    echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  done
fi
