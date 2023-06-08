echo "Benchmark system metrics"
export PYTHONPATH="${PYTHONPATH}:../."
WANDB_API_KEY=$1
REPEATS=$2
USECASE=$3

export USECASE=$USECASE
if [ $3 =  "1" ] ||  [ $3 = "3" ] || [ $3 = "4" ]; then
   round_config=(1 2 5 10)
elif [  $3 = "2" ]; then
   round_config=(1 2 4 8)
fi
# Loops trough round and number of clients configuration
for rounds in "${round_config[@]}";
do
  for client_num in {3,5,10,50}
  do
    for (( repeat = 0; repeat < $REPEATS; repeat++ ))
    do
      echo "Start repeat system metrics ${repeat} num clients ${client_num} num rounds ${rounds}"
      rm -f timelogs/tff_logs_time.txt
      tshark -q -z conv,tcp -f "tcp port 8001" -i any -B 20 > tshark_output.txt &
      tshark_pid=$!
      tshark -i any -T fields -e frame.len "tcp port 8001" -B 20 > tshark_packets.txt &
      tshark_pid2=$!
      echo "Creating single worker service"
      python worker_service.py --port 8001 --num_rounds $rounds --client_index 1 --run_repeat $repeat &
      echo "Start training"
      python tff_benchmark_gen_express.py --num_clients $client_num --num_rounds $rounds --run_repeat $repeat --network_metrics true
#      serverpid=$!
#      wait $serverpid
      pkill worker_service
      kill $tshark_pid
      kill $tshark_pid2
      project_name="usecase_${USECASE}_benchmark_rounds_${rounds}_system_metrics"
      group_name=tff_${client_num}
      run_name="run_${repeat}"
      sleep 3
      python ../scripts/log_network_stats.py --group $group_name --project $project_name --run $run_name
      echo "Repeat network metrics ${repeat} num clients ${client_num} num rounds ${rounds} complete"
      echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    done
  done
done
