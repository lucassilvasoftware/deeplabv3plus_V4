#!/bin/bash
# Submeter apenas nos NOS DE LOGIN do SDumont2nd (ex.: login.sdumont2nd.lncc.br -> sdumont2nd4-7).
# No DTN (sdumont2nd2) o Slurm nao esta disponivel: sinfo, sbatch, squeue nao funcionam.
# SDumont2nd: $HOME e $SCRATCH apontam para Lustre (/scratch ou /petrobr). Nao ha HOMEDIR dedicado como /prj. Wiki: 07-Submeter-Jobs.
#SBATCH --job-name=tcc-deeplab
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH -p lncc-h100_shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
# SBATCH --account=SIGLA   # so se for projeto PETROBRAS/Parceiros (nao e o caso de projeto FEI/LNCC)
# --time e --gpus sao obrigatorios. Doc: wiki 06, 07, 08.

# Padrao: $HOME/deeplabv3plus_V4 (ou $SCRATCH/... conforme wiki 07). Ajuste ou defina TCC_BASE_DIR antes do sbatch.
export TCC_BASE_DIR=${TCC_BASE_DIR:-$HOME/deeplabv3plus_V4}
cd "$TCC_BASE_DIR" || { echo "Erro: pasta do projeto nao encontrada: $TCC_BASE_DIR"; exit 1; }

# Ambiente Python: env em $HOME/conda-envs/tcc (ou use nome: source activate tcc). Nao ative antes do sbatch.
# Liste modulos com: module avail anaconda3
module load anaconda3/2024.02_sequana
source activate $HOME/conda-envs/tcc

# Logs com data/hora em outputs/training/training_YYYY-MM-DD_HH-MM-SS.log
python src/main.py