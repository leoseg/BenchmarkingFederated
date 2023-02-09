import pickle
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
        create_unbalanced_splits(data_path=args.data_path,label_name=args.label_name,unweight_step=args.unweighted_step)

if __name__ == '__main__':
    main()
