#!/bin/bash
#SBATCH --job-name=tcc-deeplab
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH -p lncc-h100_shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
# SBATCH --account=SIGLA   # só se for projeto PETROBRAS/Parceiros (não é o caso de projeto FEI/LNCC)
# --time e --gpus são obrigatórios. Doc: wiki 06, 07, 08. Ver README_SLURM.md.

export TCC_BASE_DIR=/scratch/psivgmp/lucas.silva8/deeplabv3plus_V4
cd /scratch/psivgmp/lucas.silva8/deeplabv3plus_V4 || { echo "Erro: pasta do projeto nao encontrada"; exit 1; }

# Ambiente Python: crie um env no cluster uma vez (ver README_SLURM.md). Depois use o nome aqui.
module load anaconda3/2024.02_sequana
source activate tcc

# Logs com data/hora em outputs/training/training_YYYY-MM-DD_HH-MM-SS.log
python src/main.py