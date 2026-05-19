#!/bin/bash
#SBATCH -p gpu_h100_4
#SBATCH --qos=qos_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=8G
#SBATCH -t 00:10:00
#SBATCH --job-name=gpu_test
#SBATCH -o ../logs/gpu_test_%j.out
#SBATCH -e ../logs/gpu_test_%j.err

echo "Running on:"
hostname

echo ""
echo "GPU INFO:"
nvidia-smi
