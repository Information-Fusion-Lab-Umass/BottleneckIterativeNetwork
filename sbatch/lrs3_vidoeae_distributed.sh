#!/bin/bash
#SBATCH -G 8                  # Number of GPUs
#SBATCH --nodes=1             # Always set to 1!
#SBATCH --gres=gpu:8          # This needs to match num GPUs.
#SBATCH --ntasks-per-node=8   # This needs to match num GPUs. default 8
#SBATCH --mem=160000           # Requested Memory
#SBATCH -p m40     # Partition
#SBATCH -t 07-00              # Job time limit
#SBATCH -o slurm-%j.out       # %j = job ID
#SBATCH -e slurm-%j.err

cd /work/pi_mfiterau_umass_edu/sidong/speech  # your own directory

eval "$(conda shell.bash hook)"
conda activate dorby  # your conda environment

export NGPU=8
export MASTER_PORT=44149

torchrun --nproc_per_node $NGPU --master_port=$MASTER_PORT scripts/run_lrs3_videoae_distributed.py --config video_ae.yaml --exp_version lrs3_e1000_single --batch_size 32