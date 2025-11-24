#!/bin/bash
#SBATCH --job-name=yolo_ssda
#SBATCH --output=yolo_ssda_%j.out
#SBATCH --error=yolo_ssda_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00

cd /ceph/home/student.aau.dk/wbilal18/ROB9/ROB9/ROB9_project

# activate your venv
source .venv/bin/activate

# run your existing training script
python -m CNN.train
