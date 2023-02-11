import pickle

import matplotlib.pyplot as plt

from utils.data_utils import create_class_balanced_partitions,create_unbalanced_splits
import argparse
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"



def main():
    parser = argparse.ArgumentParser(
        prog="partition_data.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data_path", type=str, help="path of data to load"
    )
    parser.add_argument(
        "--num_clients", type=int, help="number of clients"
    )
    parser.add_argument(
        "--unweighted_step", type=int, help="flag that show that data is that much unweighted", default=-1
    )
    parser.add_argument(
        "--label_name",type=str,help="for partitioning data",default="Condition"
    )
    args = parser.parse_args()
    if args.unweighted_step < 0:
        partitions = create_class_balanced_partitions(data_path=args.data_path, num_partitions=args.num_clients)
        with open("partitions_list","wb") as file:
            pickle.dump(partitions,file)
    else:
        class_num_df = create_unbalanced_splits(data_path=args.data_path,label_name=args.label_name,unweight_step=args.unweighted_step)
        class_num_df.plot(kind="bar", stacked=True, xlabel="Clients", ylabel="Num examples")
        plt.title = "Examples per class and per client"
        plt.show()
        percentage = args.unweighted_step * 0.05 *100
        plt.savefig(f"../plots/class_imbalance_{percentage}_clients_{args.num_clients}.png")

if __name__ == '__main__':
    main()
