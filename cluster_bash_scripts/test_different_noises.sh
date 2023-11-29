#!/bin/bash
#write me a script that start benchmark_dp_flwr with different noises defined in an array for all 4 usecases
WANDB_API_KEY=$1
NUM_REPEATS=$2
MODE=$3
echo "Current working directory: $(pwd)"
if [ $MODE = "0" ] || [ $MODE = "1" ];then
jid1=$(sbatch --parsable benchmark_dp_flwr.sh $WANDB_API_KEY $NUM_REPEATS 1 "../DataGenExpression/Alldata.csv" 2)
jid2=$(sbatch --parsable --dependency=afterany:$jid1 benchmark_dp_flwr.sh $WANDB_API_KEY $NUM_REPEATS 2 "../DataGenExpression/Alldata.csv" 2)
jid3=$(sbatch --parsable --dependency=afterany:$jid2 benchmark_dp_flwr.sh $WANDB_API_KEY $NUM_REPEATS 3 "../Dataset2/Braindata_five_classes.csv" 2)
sbatch --parsable --dependency=afterany:$jid3 benchmark_dp_flwr.sh $WANDB_API_KEY $NUM_REPEATS 4 "../Dataset2/Braindata_five_classes.csv" 2
fi
if [ $MODE = "0" ] || [ $MODE = "2" ];then
  jid4=$(sbatch --parsable benchmark_dp_tf.sh $WANDB_API_KEY $NUM_REPEATS 1 "../DataGenExpression/Alldata.csv" 2)
  jid5=$(sbatch --parsable --dependency=afterany:$jid4 benchmark_dp_tf.sh $WANDB_API_KEY $NUM_REPEATS 2 "../DataGenExpression/Alldata.csv" 2)
  jid6=$(sbatch --parsable --dependency=afterany:$jid5 benchmark_dp_tf.sh $WANDB_API_KEY $NUM_REPEATS 3 "../Dataset2/Braindata_five_classes.csv" 2)
  sbatch --parsable --dependency=afterany:$jid6 benchmark_dp_tf.sh $WANDB_API_KEY $NUM_REPEATS 4 "../Dataset2/Braindata_five_classes.csv" 2
fi
