import argparse
import subprocess

parser = argparse.ArgumentParser(description='Start multiple scripts for running a job on a cluster.')
parser.add_argument('wandb_api_key', help='Wandb API key')
parser.add_argument('num_repeats', help='Number of repeats')
parser.add_argument('system_only', help='System only flag')

args = parser.parse_args()

wandb_api_key = args.wandb_api_key
num_repeats = args.num_repeats
system_only = args.system_only

jid1 = subprocess.run(['sbatch', 'benchmark_flwr_gen_expr.sh', wandb_api_key, num_repeats, '1', '../DataGenExpression/Alldata.csv', system_only],stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True).stdout.strip()
jid2 = subprocess.run(['sbatch', '--dependency=afterany:' + jid1, 'benchmark_flwr_gen_expr.sh', wandb_api_key, num_repeats, '2', '../DataGenExpression/Alldata.csv', system_only], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True).stdout.strip()
jid3 = subprocess.run(['sbatch', '--dependency=afterany:' + jid2, 'benchmark_flwr_gen_expr.sh', wandb_api_key, num_repeats, '3', '../Dataset2/Braindata_five_classes.csv', system_only], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True).stdout.strip()
jid4 = subprocess.run(['sbatch', '--dependency=afterany:' + jid3, 'benchmark_flwr_gen_expr.sh', wandb_api_key, num_repeats, '4', '../Dataset2/Braindata_five_classes.csv', system_only], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True).stdout.strip()
jid5 = subprocess.run(['sbatch', '--dependency=afterany:' + jid4, 'benchmark_flwr_gen_expr_unweighted.sh', wandb_api_key, num_repeats, '1', '../DataGenExpression/Alldata.csv'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True).stdout.strip()
jid6 = subprocess.run(['sbatch', '--dependency=afterany:' + jid5, 'benchmark_flwr_gen_expr_unweighted.sh', wandb_api_key, num_repeats, '2', '../DataGenExpression/Alldata.csv'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True).stdout.strip()
jid7 = subprocess.run(['sbatch', '--dependency=afterany:' + jid6, 'benchmark_flwr_gen_expr_unweighted.sh', wandb_api_key, num_repeats, '3', '../Dataset2/Braindata_five_classes.csv'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True).stdout.strip()
jid8 = subprocess.run(['sbatch', '--dependency=afterany:' + jid7, 'benchmark_flwr_gen_expr_unweighted.sh', wandb_api_key, num_repeats, '4', '../Dataset2/Braindata_five_classes.csv'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True).stdout.strip()

jid9 = subprocess.run(['sbatch', 'benchmark_tff_gen_expr.sh', wandb_api_key, num_repeats, '1', '../DataGenExpression/Alldata.csv', system_only], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True).stdout.strip()
jid10 = subprocess.run(['sbatch', '--dependency=afterany:' + jid9, 'benchmark_tff_gen_expr.sh', wandb_api_key, num_repeats, '2', '../DataGenExpression/Alldata.csv', system_only], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True).stdout.strip()
jid11 = subprocess.run(['sbatch', '--dependency=afterany:' + jid10, 'benchmark_tff_gen_expr.sh', wandb_api_key, num_repeats, '3', '../Dataset2/Braindata_five_classes.csv', system_only], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True).stdout.strip()
jid12 = subprocess.run(['sbatch', '--dependency=afterany:' + jid11, 'benchmark_tff_gen_expr.sh', wandb_api_key, num_repeats, '4', '../Dataset2/Braindata_five_classes.csv', system_only], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True).stdout.strip()
jid13 = subprocess.run(['sbatch', '--dependency=afterany:' + jid12, 'benchmark_tff_gen_expr_unweighted.sh', wandb_api_key, num_repeats, '1', '../DataGenExpression/Alldata.csv'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True).stdout.strip()
jid14 = subprocess.run(['sbatch', '--dependency=afterany:' + jid13, 'benchmark_tff_gen_expr_unweighted.sh', wandb_api_key, num_repeats, '2', '../DataGenExpression/Alldata.csv'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True).stdout.strip()
jid15 = subprocess.run(['sbatch', '--dependency=afterany:' + jid14, 'benchmark_tff_gen_expr_unweighted.sh', wandb_api_key, num_repeats, '3', '../Dataset2/Braindata_five_classes.csv'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True).stdout.strip()
jid16 = subprocess.run(['sbatch', '--dependency=afterany:' + jid15, 'benchmark_tff_gen_expr_unweighted.sh', wandb_api_key, num_repeats, '4', '../Dataset2/Braindata_five_classes.csv'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True).stdout.strip()