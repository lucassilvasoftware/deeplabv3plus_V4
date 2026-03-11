#!/bin/bash
# Submeter apenas nos NOS DE LOGIN do SDumont2nd (ex.: login.sdumont2nd.lncc.br -> sdumont2nd4-7).
# No DTN (sdumont2nd2) o Slurm nao esta disponivel: sinfo, sbatch, squeue nao funcionam.
# SDumont2nd: $HOME e $SCRATCH apontam para Lustre. Wiki: 06, 07, 08 (Submeter Jobs).
#SBATCH --job-name=tcc-deeplab
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH -p lncc-h100_shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
# SBATCH --account=SIGLA   # somente se projeto PETROBRAS/Parceiros

# Raiz do projeto no cluster. Defina antes do sbatch se diferente (ex.: export TCC_BASE_DIR=$SCRATCH/tcc-v2-DeeplabV3Plus).
# Necessário: processed_dataset/dataset_registry.yaml e dados em ROOT/datasets_brutos/ (ou path indicado no registry).
export TCC_BASE_DIR=${TCC_BASE_DIR:-$HOME/tcc-v2-DeeplabV3Plus}
cd "$TCC_BASE_DIR" || { echo "Erro: pasta do projeto nao encontrada: $TCC_BASE_DIR"; exit 1; }

# Salvar job id para cancelar depois (opcional: cancelar com scancel $(cat outputs/training/slurm_job_id.txt))
mkdir -p outputs/training
echo $SLURM_JOB_ID > outputs/training/slurm_job_id.txt

# Ambiente Python (ajuste o nome do env se necessário)
module load anaconda3/2024.02_sequana
source activate $HOME/conda-envs/tcc

# Treino: TEST_MODE=False, batch 24, LR 2e-4, early stop 12. Log detalhado em outputs/training/training_*.log
# Acompanhar progresso: tail -f slurm_*.out ou grep PROGRESS outputs/training/training_*.log
python src/main.py