from evaluation_utils import evaluate_model, load_test_data_for_evaluation
#import wandb
from config import configs
# import argparse
# parser = argparse.ArgumentParser(
#     prog="evaluate_fl_model.py",
#     formatter_class=argparse.RawDescriptionHelpFormatter,
# )
# CLI arguments are project name and group name, and run repeat
# parser.add_argument("--project", type=str, help="wandb project name")
# parser.add_argument("--group", type=str, help="wandb group name")
# parser.add_argument("--run_repeat", type=int, help="number of run with same config", default=1)

#args = parser.parse_args()
X_test, y_test = load_test_data_for_evaluation(0)
path_to_tff_weights = f"../TensorflowFederated/tff_weights_{0}.h5"
path_to_flwr_weights = f"../Flower/flwr_weights_{0}.h5"
#wandb.init(project=args.project, group=f"flwr_{args.group}", name=f"run_{args.run_repeat}",config=configs)
score = evaluate_model(path_to_flwr_weights, X_test,y_test)
print(score)
# wandb.log(score)
# wandb.finish()
#wandb.init(project=args.project, group=f"tff_{args.group}", name=f"run_{args.run_repeat}",config=configs)
score = evaluate_model(path_to_tff_weights, X_test,y_test)
print(score)
#wandb.log(score)
#wandb.finish()
