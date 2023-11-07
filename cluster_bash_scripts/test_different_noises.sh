#!/bin/bash
#write me a script that start benchmark_dp_flwr with different noises defined in an array for all 4 usecases
WANDB_API_KEY=$1
NUM_REPEATS=$2
echo "Current working directory: $(pwd)"
jid1=$(sbatch --parsable benchmark_dp_flwr.sh $WANDB_API_KEY $NUM_REPEATS 1 "../DataGenExpression/Alldata.csv" 2)
jid2=$(sbatch --parsable --dependency=afterany:$jid1 benchmark_dp_flwr.sh $WANDB_API_KEY $NUM_REPEATS 2 "../DataGenExpression/Alldata.csv" 2)
jid3=$(sbatch --parsable --dependency=afterany:$jid2 benchmark_dp_flwr.sh $WANDB_API_KEY $NUM_REPEATS 3 "../Dataset2/Braindata_five_classes.csv" 2)
sbatch --parsable --dependency=afterany:$jid3 benchmark_dp_flwr.sh $WANDB_API_KEY $NUM_REPEATS 4 "../Dataset2/Braindata_five_classes.csv" 2
