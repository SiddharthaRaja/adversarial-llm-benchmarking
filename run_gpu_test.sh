#!/bin/bash
#SBATCH -p gpu_h100_4
#SBATCH --qos=qos_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH -t 00:10:00
#SBATCH --job-name=gpu_python_test
#SBATCH -o ../logs/gpu_python_test_%j.out
#SBATCH -e ../logs/gpu_python_test_%j.err

module load anaconda3-2021.05-gcc-8.5.0-tzdgetu

source /apps/spack/opt/spack/linux-rocky8-zen/gcc-8.5.0/anaconda3-2021.05-tzdgetua6vg6dstl26ufvg42fgcrjv2u/etc/profile.d/conda.sh

conda activate tcga_env

export PYTHONPATH=/scratch/spanda/py_packages_classification:$PYTHONPATH

export HF_HOME=/scratch/spanda/tcga_brca_data_classification/eval_utils/hf_cache

python ../models/gpu_test.py
