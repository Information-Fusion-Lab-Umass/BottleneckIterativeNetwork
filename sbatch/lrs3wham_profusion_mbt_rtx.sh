#!/bin/bash
#SBATCH -G 6                  # Number of GPUs
#SBATCH --nodes=1             # Always set to 1!
#SBATCH --gres=gpu:6          # This needs to match num GPUs.
#SBATCH --ntasks-per-node=6   # This needs to match num GPUs. default 8
#SBATCH --mem=300000           # Requested Memory
#SBATCH -p gypsum-rtx8000     # Partition
#SBATCH -t 07-00              # Job time limit
#SBATCH -o slurm-%j.out       # %j = job ID

cd /work/pi_mfiterau_umass_edu/sidong/speech  # your own directory

eval "$(conda shell.bash hook)"
conda activate dorby  # your conda environment

export NGPU=6
export MASTER_PORT=44147

torchrun --nproc_per_node $NGPU --master_port=$MASTER_PORT scripts/run_lrs3wham_prombt.py --config avlit.yaml --exp_version prombt12_wham_reg0_f32_rtx --recover_weight 0 --batch_size 16  --num_blocks 12 --fusion_dim 32