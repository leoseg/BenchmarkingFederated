import pickle
import matplotlib.pyplot as plt
from utils.config import configs
from utils.data_utils import create_class_balanced_partitions,create_unbalanced_splits,load_data,preprocess_data
import argparse
import os
# Calls data partition function depending on benchmark configuration and dumps resulting list of list of row indices
# for each client
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
def main():
    parser = argparse.ArgumentParser(
        prog="partition_data.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data_path", type=str, help="path of data to load",default=configs.get("data_path")
    )
    parser.add_argument(
        "--num_clients", type=int, help="number of clients"
    )
    parser.add_argument(
        "--unweighted_step", type=int, help="flag that show that data is that much unweighted", default=-1
    )
    parser.add_argument(
        "--label_name",type=str,help="for partitioning data",default=configs["label"]
    )
    args = parser.parse_args()

    df = load_data(data_path=args.data_path)
    #df = preprocess_data(df)
    # If no unweighted_step created balance partitions
    if args.unweighted_step < 0:
        partitions = create_class_balanced_partitions(df, num_partitions=args.num_clients)
        with open("partitions_list","wb") as file:
            pickle.dump(partitions,file)
    else:
        class_num_df,partitions = create_unbalanced_splits(df,label_name=args.label_name,unweight_step=args.unweighted_step)
        # Create charts and dictionary of number of samples per class and client and saves them
        class_num_df.plot(kind="bar", stacked=True, xlabel="Clients", ylabel="Num examples")
        class_num_df.to_csv(f"partitions_dict_{args.unweighted_step}.csv")
        plt.title = "Examples per class and per client"
        percentage = args.unweighted_step * 0.05 *100
        plt.savefig(f"../plots/class_imbalance_{percentage}_clients_{args.num_clients}.png")
        with open("partitions_list","wb") as file:
            pickle.dump(partitions,file)





if __name__ == '__main__':
    main()
