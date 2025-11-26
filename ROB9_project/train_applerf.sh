#!/bin/bash
#SBATCH --job-name=appleRF
#SBATCH --partition=l4
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=applerf_%j.out
#SBATCH --error=applerf_%j.err

# Go to your project
cd /ceph/home/student.aau.dk/wbilal18/ROB9/ROB9/ROB9_project

# Activate venv
source .venv/bin/activate

# Run your AppleRF training (n, s, m)
python -m CNN.trainRF
