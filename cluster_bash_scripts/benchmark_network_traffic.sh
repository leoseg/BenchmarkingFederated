WANDB_API_KEY=$1
cd ..
cd Flower || exit
# benchmark network traffic for all usecases for flower
#for USECASE in {2,3};
#do
#  bash flwr_network_bandwidth.sh $WANDB_API_KEY 1 $USECASE
#done
cd ..
cd TensorflowFederated || exit
# benchmark network traffic for all usecases for tff
for USECASE in {2,3};
do
  bash tff_network_bandwidth.sh $WANDB_API_KEY 1 $USECASE
done