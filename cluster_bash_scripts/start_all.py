import argparse
import subprocess

parser = argparse.ArgumentParser(description='Start multiple scripts for running a job on a cluster.')
parser.add_argument('wandb_api_key', help='Wandb API key')
parser.add_argument('num_repeats', help='Number of repeats')
parser.add_argument('system_only', help='System only flag')

args = parser.parse_args()

wandb_api_key = args.wandb_api_key
num_repeats = args.num_repeats
system_only = args.system_only_flag

jid1 = subprocess.run(['sbatch', 'benchmark_flwr_gen_expr.sh', wandb_api_key, num_repeats, '1', '../DataGenExpression/Alldata.csv', system_only], capture_output=True, text=True)
jid2 = subprocess.run(['sbatch', '--dependency=afterany:' + jid1, 'benchmark_flwr_gen_expr.sh', wandb_api_key, num_repeats, '2', '../DataGenExpression/Alldata.csv', system_only], capture_output=True, text=True)
jid3 = subprocess.run(['sbatch', '--dependency=afterany:' + jid2, 'benchmark_flwr_gen_expr.sh', wandb_api_key, num_repeats, '3', '../Dataset2/Braindata_five_classes.csv', system_only], capture_output=True, text=True)
jid4 = subprocess.run(['sbatch', '--dependency=afterany:' + jid3, 'benchmark_flwr_gen_expr.sh', wandb_api_key, num_repeats, '4', '../Dataset2/Braindata_five_classes.csv', system_only], capture_output=True, text=True)
jid5 = subprocess.run(['sbatch', '--dependency=afterany:' + jid4, 'benchmark_flwr_gen_expr_unweighted.sh', wandb_api_key, num_repeats, '1', '../DataGenExpression/Alldata.csv'], capture_output=True, text=True)
jid6 = subprocess.run(['sbatch', '--dependency=afterany:' + jid5, 'benchmark_flwr_gen_expr_unweighted.sh', wandb_api_key, num_repeats, '2', '../DataGenExpression/Alldata.csv'], capture_output=True, text=True)
jid7 = subprocess.run(['sbatch', '--dependency=afterany:' + jid6, 'benchmark_flwr_gen_expr_unweighted.sh', wandb_api_key, num_repeats, '3', '../Dataset2/Braindata_five_classes.csv'], capture_output=True, text=True)
jid8 = subprocess.run(['sbatch', '--dependency=afterany:' + jid7, 'benchmark_flwr_gen_expr_unweighted.sh', wandb_api_key, num_repeats, '4', '../Dataset2/Braindata_five_classes.csv'], capture_output=True, text=True)

jid9 = subprocess.run(['sbatch', 'benchmark_tff_gen_expr.sh', wandb_api_key, num_repeats, '1', '../DataGenExpression/Alldata.csv', system_only], capture_output=True, text=True)
jid10 = subprocess.run(['sbatch', '--dependency=afterany:' + jid9, 'benchmark_tff_gen_expr.sh', wandb_api_key, num_repeats, '2', '../DataGenExpression/Alldata.csv', system_only], capture_output=True, text=True)
jid11 = subprocess.run(['sbatch', '--dependency=afterany:' + jid10, 'benchmark_tff_gen_expr.sh', wandb_api_key, num_repeats, '3', '../Dataset2/Braindata_five_classes.csv', system_only], capture_output=True, text=True)
jid12 = subprocess.run(['sbatch', '--dependency=afterany:' + jid11, 'benchmark_tff_gen_expr.sh', wandb_api_key, num_repeats, '4', '../Dataset2/Braindata_five_classes.csv', system_only], capture_output=True, text=True)
jid13 = subprocess.run(['sbatch', '--dependency=afterany:' + jid12, 'benchmark_tff_gen_expr_unweighted.sh', wandb_api_key, num_repeats, '1', '../DataGenExpression/Alldata.csv'], capture_output=True, text=True)
jid14 = subprocess.run(['sbatch', '--dependency=afterany:' + jid13, 'benchmark_tff_gen_expr_unweighted.sh', wandb_api_key, num_repeats, '2', '../DataGenExpression/Alldata.csv'], capture_output=True, text=True)
jid15 = subprocess.run(['sbatch', '--dependency=afterany:' + jid14, 'benchmark_tff_gen_expr_unweighted.sh', wandb_api_key, num_repeats, '3', '../Dataset2/Braindata_five_classes.csv'], capture_output=True, text=True)
jid16 = subprocess.run(['sbatch', '--dependency=afterany:' + jid15, 'benchmark_tff_gen_expr_unweighted.sh', wandb_api_key, num_repeats, '4', '../Dataset2/Braindata_five_classes.csv'], capture_output=True, text=True)