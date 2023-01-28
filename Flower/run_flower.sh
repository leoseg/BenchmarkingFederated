#!/bin/bash

echo "Starting server"
export PYTHONPATH="${PYTHONPATH}:../."
echo $PYTHONPATH
python server.py &
sleep 3  # Sleep for 3s to give the server enough time to start
python client.py
#for i in `seq 0`; do
#    echo "Starting client $i"
#    python client.py &
#done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait