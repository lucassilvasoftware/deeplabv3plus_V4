#!/bin/bash
#SBATCH --job-name=tcc-deeplab
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH -p lncc-h100_shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
# SBATCH --account=SIGLA_DO_PROJETO
# ↑ Descomente e preencha se for projeto PETROBRAS/Parceiros (obrigatório).
#   Liste com: sacctmgr list user $USER -s format=partition%20,account,MaxJobs,MaxWall

# Raiz do projeto no Lustre (SDumont2nd: $HOME e $SCRATCH apontam para Lustre)
export TCC_BASE_DIR=${SCRATCH:-$HOME}/tcc-v2-DeeplabV3Plus
cd $TCC_BASE_DIR

# Conda: carregar e ativar DENTRO do script (não ativar antes de sbatch)
# Ajuste o módulo conforme disponível no cluster: module avail anaconda
module load anaconda3/2024.02_sequana
source activate $SCRATCH/conda-tcc

python src/main.py
