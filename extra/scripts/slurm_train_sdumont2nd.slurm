#!/bin/bash
# Submeter apenas nos NOS DE LOGIN (wiki 07). No DTN (sdumont2nd2) Slurm nao esta disponivel.
# Manual: https://github.com/lncc-sered/manual-sdumont/wiki
# Regras: (06) fila GPU exige --time e --gpus; (07) copiar projeto/dados para SCRATCH se Home nao for acessivel nos nos de execucao; (09) nao ativar conda antes do sbatch.
#SBATCH --job-name=tcc-deeplab
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH -p lncc-h100_shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
# SBATCH --account=SIGLA   # somente se projeto PETROBRAS/Parceiros

# Raiz do projeto. Se os nos de computacao NAO montam Home, use SCRATCH: export TCC_BASE_DIR=$SCRATCH/tcc-v2-DeeplabV3Plus antes do sbatch.
# Necessario: processed_dataset/dataset_registry.yaml e dados (images-tiff, labels, splits) em TCC_BASE_DIR ou paths do registry.
export TCC_BASE_DIR=${TCC_BASE_DIR:-$HOME/tcc-v2-DeeplabV3Plus}
cd "$TCC_BASE_DIR" || { echo "Erro: pasta do projeto nao encontrada: $TCC_BASE_DIR"; exit 1; }

# Salvar job id para cancelar depois (opcional: cancelar com scancel $(cat outputs/training/slurm_job_id.txt))
mkdir -p outputs/training
echo $SLURM_JOB_ID > outputs/training/slurm_job_id.txt

# Ambiente Python (wiki 09: criar/ativar so dentro do script; recomendado env em $SCRATCH/conda-envs/tcc se Home nao acessivel nos nos)
module load anaconda3/2024.02_sequana
source activate ${TCC_CONDA_ENV:-$HOME/conda-envs/tcc}

# Treino: TEST_MODE=False, batch 24, LR 2e-4, early stop 12. Log detalhado em outputs/training/training_*.log
# Acompanhar progresso: tail -f slurm_*.out ou grep PROGRESS outputs/training/training_*.log
python src/main.py