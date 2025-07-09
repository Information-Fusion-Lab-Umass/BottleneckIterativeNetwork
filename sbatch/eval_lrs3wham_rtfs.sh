#!/bin/bash
#SBATCH -G 2                  # Number of GPUs
#SBATCH --nodes=1             # Always set to 1!
#SBATCH --gres=gpu:2          # This needs to match num GPUs.
#SBATCH --ntasks-per-node=2   # This needs to match num GPUs. default 8
#SBATCH --mem=100000           # Requested Memory
#SBATCH -p gpu     # Partition
#SBATCH --constraint=a100
#SBATCH -t 02-00              # Job time limit
#SBATCH -o slurm-%j.out       # %j = job ID

cd /project/pi_mfiterau_umass_edu/sidong/speech  # your own directory

eval "$(conda shell.bash hook)"
conda activate dorby  # your conda environment

export NGPU=2
export MASTER_PORT=44127

torchrun --nproc_per_node $NGPU --master_port=$MASTER_PORT scripts/eval_lrs3wham_rtfs.py --config rtfs12.yaml --exp_version lrs3wham_b12_a100 --batch_size 8 --recover_weight 0