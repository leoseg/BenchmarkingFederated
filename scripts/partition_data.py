import pickle
from utils.data_utils import create_class_balanced_partitions,create_unbalanced_splits
import argparse
import os
from utils.config import path_to_partitionlist
from utils.config import jobid
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
    args = parser.parse_args()
    if args.unweighted_percentage < 0:
        partitions = create_class_balanced_partitions(data_path=args.data_path, num_partitions=args.num_clients)
        with open(path_to_partitionlist,"wb") as file:
            pickle.dump(partitions,file)
    else:
        create_unbalanced_splits(data_path=args.data_path,label_name=args.label_name,unweight_step=args.unweighted_step,job_id=jobid)

if __name__ == '__main__':
    main()
