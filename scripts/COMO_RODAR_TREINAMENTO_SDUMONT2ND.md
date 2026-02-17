# SDumont2nd – comandos

## Conexão / projeto no cluster
```bash
# Onde está o projeto (HOME ou SCRATCH)
echo $HOME
echo $SCRATCH
cd $HOME   # ou $SCRATCH
# clone/copie o repo, ex: $HOME/deeplabv3plus_V4
```

## Ambiente Conda (uma vez)
```bash
module avail anaconda3
module load anaconda3/2024.10   # ou versão disponível
conda create --prefix $HOME/conda-envs/tcc python=3.10
source activate $HOME/conda-envs/tcc
pip install torch torchvision
pip install segmentation-models-pytorch
# demais do requirements.txt
conda deactivate
```

## Ajustar script
- `scripts/slurm_train_sdumont2nd.slurm`: conferir `TCC_BASE_DIR`, `module load anaconda3`, `source activate`.
- Se PETROBRAS: descomentar `#SBATCH --account=SIGLA`.

## Submeter
```bash
cd ~/deeplabv3plus_V4   # ou $TCC_BASE_DIR
sbatch scripts/slurm_train_sdumont2nd.slurm
```

## Ver filas
```bash
sinfo
sacctmgr list user $USER -s format=partition%20,MaxJobs,MaxSubmit,MaxNodes,MaxCPUs,MaxWall
```

## Acompanhar
```bash
squeue -u $USER
squeue --start -u $USER
scancel JOBID
# logs: slurm_JOBID.out, slurm_JOBID.err; treino em outputs/training/
```
