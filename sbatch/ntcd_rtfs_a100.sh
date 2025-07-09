#!/bin/bash
#SBATCH -G 8                  # Number of GPUs
#SBATCH --nodes=1             # Always set to 1!
#SBATCH --gres=gpu:8          # This needs to match num GPUs.
#SBATCH --ntasks-per-node=8   # This needs to match num GPUs. default 8
#SBATCH --mem=300000           # Requested Memory
#SBATCH -p gpu-preempt     # Partition
#SBATCH --constraint=a100
#SBATCH -t 07-00              # Job time limit
#SBATCH -o slurm-%j.out       # %j = job ID

cd /work/pi_mfiterau_umass_edu/sidong/speech  # your own directory

eval "$(conda shell.bash hook)"
conda activate dorby  # your conda environment

export NGPU=8
export MASTER_PORT=44137

torchrun --nproc_per_node $NGPU --master_port=$MASTER_PORT scripts/run_ntcd_rtfs.py --config rtfs12.yaml --exp_version ntcd_b12_a100 --batch_size 4 --recover_weight 0