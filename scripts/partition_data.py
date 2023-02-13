import pickle
from functools import reduce

import matplotlib.pyplot as plt
import pandas as pd
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
        '--data_pathes', nargs='+', default=[],help="path multiple datapathes"
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
        if args.data_path:
            datapathes = [args.data_path]
        else:
            datapathes = args.data_pathes
        class_num_dfs = []
        list_partition_dfs = []
        for datapath in datapathes:
            class_num_df,partition_dfs = create_unbalanced_splits(data_path=datapath,label_name=args.label_name,unweight_step=args.unweighted_step)
            class_num_dfs.append(class_num_df)
            list_partition_dfs.append(partition_dfs)
        partition_dfs = reduce(lambda  x,y: [ pd.concat([xi,yi]) for xi,yi in zip(x,y)],list_partition_dfs)
        class_num_df = reduce(lambda x, y: x.add(y, fill_value=0), class_num_dfs)
        class_num_df.plot(kind="bar", stacked=True, xlabel="Clients", ylabel="Num examples")
        plt.title = "Examples per class and per client"
        percentage = args.unweighted_step * 0.05 *100
        plt.savefig(f"../plots/class_imbalance_{percentage}_clients_{args.num_clients}.png")
        for count,df in  enumerate(partition_dfs):
            df.to_csv(f"partition_{count}.csv")




if __name__ == '__main__':
    main()
