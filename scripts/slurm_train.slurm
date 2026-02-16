#!/bin/bash
#SBATCH --job-name=tcc-deeplab
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH -p sequana_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem=32G

# Raiz do projeto no SCRATCH (ajuste NOME_DO_PROJETO e SEU_USUARIO)
export TCC_BASE_DIR=/scratch/NOME_DO_PROJETO/SEU_USUARIO/tcc-v2-DeeplabV3Plus
cd $TCC_BASE_DIR

# Conda: carregar e ativar DENTRO do script (não ativar antes de sbatch)
module load anaconda3/2024.02_sequana
source activate $SCRATCH/conda-tcc

python src/main.py
