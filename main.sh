#!/bin/bash

#SBATCH -p gpu --gres=gpu:2
#SBATCH -n 4
#SBATCH -t 48:00:00
#SBATCH --mem=16G
#SBATCH -J Job0ClinicalFineTuning
#SBATCH -o Job0ClinicalFineTuning-%j.out
#SBATCH -e Job0ClinicalFineTuning-%j.out

export PYTHONUNBUFFERED=TRUE

python /oscar/home/hzheng29/data/hzheng29/my_clinical_fine_tuning/main.py