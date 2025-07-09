#!/bin/bash
#SBATCH -G 4                  # Number of GPUs
#SBATCH --nodes=1             # Always set to 1!
#SBATCH --gres=gpu:4          # This needs to match num GPUs.
#SBATCH --ntasks-per-node=4   # This needs to match num GPUs. default 8
#SBATCH --mem=160000           # Requested Memory
#SBATCH -p gypsum-2080ti     # Partition
#SBATCH -t 07-00              # Job time limit
#SBATCH -o slurm-%j.out       # %j = job ID
#SBATCH -e slurm-%j.err

cd /project/pi_mfiterau_umass_edu/sidong/speech  # your own directory

eval "$(conda shell.bash hook)"
conda activate dorby  # your conda environment

export NGPU=4
export MASTER_PORT=44149

torchrun --nproc_per_node $NGPU --master_port=$MASTER_PORT scripts/generate_lrs3_videolatent.py --config video_ae.yaml --exp_version lrs3_e500 --batch_size 32