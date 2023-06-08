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
      rm -f timelogs/flw_logs_time.txt
      tshark -q -z conv,tcp -f "tcp port 8150" -i any -B 20 > tshark_output.txt &
      tshark_pid=$!
      tshark -i any -T fields -e frame.len "tcp port 8150" -B 20 > tshark_packets.txt &
      tshark_pid2=$!
      echo "Creating server"
      python server.py --run_repeat $repeat --num_clients 1 --num_rounds $rounds --network_metrics true &
      serverpid=$!
      sleep 10
      echo "Start client"
      python client.py --client_index 1 --run_repeat $repeat
      project_name="usecase_${USECASE}_benchmark_rounds_${rounds}_system_metrics"
      group_name=flwr_${client_num}
      run_name="run_${repeat}"
      wait $serverpid
      kill $tshark_pid
      kill $tshark_pid2
      python ../scripts/log_network_stats.py --group $group_name --project $project_name --run $run_name
      echo "Repeat network metrics ${repeat} num clients ${client_num} num rounds ${rounds} complete"
      echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    done
  done
done