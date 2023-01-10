#!/bin/bash
#SBATCH --nodes=2
#SBATCH --job-name=training_unsupervised
#SBATCH --partition=clara
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=leoseeger16@gmail.com

module load CUDAcore/11.2.1
module load cuDNN/8.1.0.77-CUDA-11.2.1
module load Anaconda3
singularity instance.start -B output/dir/outside/:/output/ --net --network-args "portmap=8080:80/tcp" docker://nginx web2
