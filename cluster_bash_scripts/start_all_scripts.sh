#!/bin/bash
#write me a script that starts all the scripts in the cluster_bash_scripts folder and has as input the systems_only and wandapi key and number of repeats
WANDB_API_KEY=$1
NUM_REPEATS=$2
SYSTEM_ONLY=$3
MODE=$4
echo "Current working directory: $(pwd)"
if [ $MODE = "1" ] || [ $MODE = "2" ];then
  jid1=$(sbatch --parsable benchmark_flwr_gen_expr.sh  $WANDB_API_KEY $NUM_REPEATS 1 "../DataGenExpression/Alldata.csv" $SYSTEM_ONLY)
  jid2=$(sbatch --parsable --dependency=afterany:$jid1 benchmark_flwr_gen_expr.sh $WANDB_API_KEY $NUM_REPEATS 2 "../DataGenExpression/Alldata.csv" $SYSTEM_ONLY)
  jid3=$(sbatch --parsable --dependency=afterany:$jid2 benchmark_flwr_gen_expr.sh $WANDB_API_KEY $NUM_REPEATS 3 "../Dataset2/Braindata_five_classes.csv" $SYSTEM_ONLY)
  jid4=$(sbatch --parsable --dependency=afterany:$jid3 benchmark_flwr_gen_expr.sh $WANDB_API_KEY $NUM_REPEATS 4 "../Dataset2/Braindata_five_classes.csv" $SYSTEM_ONLY)
fi
if [ $MODE =  "1" ];then
  jid5=$(sbatch --parsable benchmark_flwr_gen_expr_unweighted.sh $WANDB_API_KEY $NUM_REPEATS 1 "../DataGenExpression/Alldata.csv")
fi
if [ $MODE = "2" ];then
  jid5=$(sbatch --parsable --dependency=afterany:$jid4 benchmark_flwr_gen_expr_unweighted.sh $WANDB_API_KEY $NUM_REPEATS 1 "../DataGenExpression/Alldata.csv")
fi
if [ $MODE = "1" ] || [ $MODE = "2" ];then
  jid6=$(sbatch --parsable --dependency=afterany:$jid5 benchmark_flwr_gen_expr_unweighted.sh $WANDB_API_KEY $NUM_REPEATS 2 "../DataGenExpression/Alldata.csv")
  jid7=$(sbatch --parsable --dependency=afterany:$jid6 benchmark_flwr_gen_expr_unweighted.sh $WANDB_API_KEY $NUM_REPEATS 3 "../Dataset2/Braindata_five_classes.csv")
  sbatch --dependency=afterany:$jid7 benchmark_flwr_gen_expr_unweighted.sh $WANDB_API_KEY $NUM_REPEATS 4 "../Dataset2/Braindata_five_classes.csv"
fi
# do the same for tff
if [ $MODE = "1" ] || [ $MODE = "2" ];then
  jid8=$(sbatch --parsable benchmark_tff_gen_expr.sh $WANDB_API_KEY $NUM_REPEATS 1 "../DataGenExpression/Alldata.csv" $SYSTEM_ONLY)
  jid9=$(sbatch --parsable --dependency=afterany:$jid8 benchmark_tff_gen_expr.sh $WANDB_API_KEY $NUM_REPEATS 2 "../DataGenExpression/Alldata.csv" $SYSTEM_ONLY)
  jid10=$(sbatch --parsable --dependency=afterany:$jid9 benchmark_tff_gen_expr.sh $WANDB_API_KEY $NUM_REPEATS 3 "../Dataset2/Braindata_five_classes.csv" $SYSTEM_ONLY)
  jid11=$(sbatch --parsable --dependency=afterany:$jid10 benchmark_tff_gen_expr.sh $WANDB_API_KEY $NUM_REPEATS 4 "../Dataset2/Braindata_five_classes.csv" $SYSTEM_ONLY)
fi
if [ $MODE = "1" ];then
  jid12=$(sbatch --parsable benchmark_tff_gen_expr_unweighted.sh $WANDB_API_KEY $NUM_REPEATS 1 "../DataGenExpression/Alldata.csv")
fi
if [ $MODE = "2" ];then
  jid12=$(sbatch --parsable --dependency=afterany:$jid11 benchmark_tff_gen_expr_unweighted.sh $WANDB_API_KEY $NUM_REPEATS 1 "../DataGenExpression/Alldata.csv")
fi
if  [ $MODE = "1" ] || [ $MODE = "2" ];then
  jid13=$(sbatch --parsable --dependency=afterany:$jid12 benchmark_tff_gen_expr_unweighted.sh $WANDB_API_KEY $NUM_REPEATS 2 "../DataGenExpression/Alldata.csv")
  jid14=$(sbatch --parsable --dependency=afterany:$jid13 benchmark_tff_gen_expr_unweighted.sh $WANDB_API_KEY $NUM_REPEATS 3 "../Dataset2/Braindata_five_classes.csv")
  sbatch --dependency=afterany:$jid14 benchmark_tff_gen_expr_unweighted.sh $WANDB_API_KEY $NUM_REPEATS 4 "../Dataset2/Braindata_five_classes.csv"
fi
