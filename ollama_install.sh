#!/bin/bash
#SBATCH -p gpu_h100_4
#SBATCH --qos=qos_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH --job-name=ollama_install
#SBATCH -o ../logs/ollama_install_%j.out
#SBATCH -e ../logs/ollama_install_%j.err

cd ../ollama

bash install_ollama.sh

ollama --version
