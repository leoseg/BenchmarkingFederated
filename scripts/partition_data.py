import pickle

from utils.data_utils import create_class_balanced_partitions
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
    args = parser.parse_args()
    partitions = create_class_balanced_partitions(data_path=args.data_path, num_partitions=args.num_clients)
    with open("partitions_list","wb") as file:
        pickle.dump(partitions,file)


if __name__ == '__main__':
    main()
