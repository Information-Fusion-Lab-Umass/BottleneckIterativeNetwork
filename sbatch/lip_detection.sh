#!/bin/bash
#SBATCH --nodes=1             # Always set to 1!
#SBATCH --cpus-per-task=20
#SBATCH --mem=80000           # Requested Memory
#SBATCH -p cpu-long           # Partition
#SBATCH -t 07-00              # Job time limit
#SBATCH -o slurm-%j.out       # %j = job ID
#SBATCH -e slurm-%j.err

cd /project/pi_mfiterau_umass_edu/sidong/speech/dataset/lrs3/script

eval "$(conda shell.bash hook)"
conda activate dorby

python process_video.py