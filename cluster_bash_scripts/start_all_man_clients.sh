#!/bin/bash
WANDB_API_KEY=$1
NUM_REPEATS=$2
SYSTEM_ONLY=$3
CLIENTS=$4
#jjid1=$(sbatch --parsable benchmark_many_clients.sh $WANDB_API_KEY $NUM_REPEATS 1 "../DataGenExpression/Alldata.csv" $SYSTEM_ONLY "TFF" $CLIENTS)
#jiid2=$(sbatch --parsable --dependency=afterany:$jjid1 benchmark_many_clients.sh $WANDB_API_KEY $NUM_REPEATS 2 "../DataGenExpression/Alldata.csv" $SYSTEM_ONLY "TFF" $CLIENTS)
#jjid3=$(sbatch --parsable --dependency=afterany:$jiid2 benchmark_many_clients.sh $WANDB_API_KEY $NUM_REPEATS 3 "../Dataset2/Braindata_five_classes.csv" $SYSTEM_ONLY "TFF" $CLIENTS)
#sbatch --parsable --dependency=afterany:$jjid3 benchmark_many_clients.sh $WANDB_API_KEY $NUM_REPEATS 4 "../Dataset2/Braindata_five_classes.csv" $SYSTEM_ONLY "TFF" $CLIENTS
# do the same for flwr
#jiid4=$(sbatch --parsable benchmark_many_clients.sh $WANDB_API_KEY $NUM_REPEATS 1 "../DataGenExpression/Alldata.csv" $SYSTEM_ONLY "FLWR" $CLIENTS)
#jiid5=$(sbatch --parsable --dependency=afterany:$jiid4 benchmark_many_clients.sh $WANDB_API_KEY $NUM_REPEATS 2 "../DataGenExpression/Alldata.csv" $SYSTEM_ONLY "FLWR" $CLIENTS)
jiid5=$(sbatch --parsable benchmark_many_clients.sh $WANDB_API_KEY $NUM_REPEATS 2 "../DataGenExpression/Alldata.csv" $SYSTEM_ONLY "FLWR" $CLIENTS)
jiid6=$(sbatch --parsable --dependency=afterany:$jiid5 benchmark_many_clients.sh $WANDB_API_KEY $NUM_REPEATS 3 "../Dataset2/Braindata_five_classes.csv" $SYSTEM_ONLY "FLWR" $CLIENTS)
sbatch --parsable --dependency=afterany:$jiid6 benchmark_many_clients.sh $WANDB_API_KEY $NUM_REPEATS 4 "../Dataset2/Braindata_five_classes.csv" $SYSTEM_ONLY "FLWR" $CLIENTS
