import argparse
from utils.system_utils import draw_group_plot_of_df,read_system_logs
import wandb
parser = argparse.ArgumentParser(
        prog="mem_data_to_wandb.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
parser.add_argument(
    "--logs_path",type=str,help="path to logs"
)
parser.add_argument(
    "--run_name",type=str,help="name of run"
)
parser.add_argument(
    "--group_name",type=str,help="name of group"
)
parser.add_argument(
    "--project_name",type=str,help="name of project"
)
parser.add_argument(
    "--memory_type",type=str,help="type of memory measured"
)

args = parser.parse_args()
read_system_logs(args.logs_path,project_name=args.project_name,group_name=args.group_name,run_name=args.run_name,memory_type=args.memory_type)

