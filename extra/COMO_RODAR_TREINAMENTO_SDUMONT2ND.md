# Treino no Santos Dumont (SDumont) — conforme manual LNCC

Referência: [manual-sdumont (wiki)](https://github.com/lncc-sered/manual-sdumont/wiki).

---

## Regras extraídas do manual

1. **Home vs Scratch (wiki 02 – Características)**  
   - **Scratch** (`/scratch/PROJETO/USUARIO`, variável `$SCRATCH`): área para **tudo** que o job usa em execução (script, executável, dados de entrada/saída).  
   - **Home** (`/prj/...`): em parte das configurações do SDumont, **não é acessível nos nós de computação**, apenas nos nós de login.  
   - **Procedimento recomendado (wiki 07 – Submeter Jobs):** copiar script, código, dados e dependências para o **SCRATCH** e rodar o job a partir daí. Assim o job sempre encontra os arquivos, em qualquer versão do cluster.

2. **Submissão (wiki 07)**  
   - Submissão só a partir dos **nós de login** (não no DTN).  
   - Parâmetros SLURM no script devem usar `#SBATCH`.  
   - Script deve começar com `#!/bin/bash`.

3. **Filas GPU (wiki 06)**  
   - Para fila GPU é **obrigatório** informar número de GPUs (ex.: `--gpus=1` ou `--gpus-per-node=1`). Caso contrário o job pode ficar pendente com REASON `QOSMinGRES`.  
   - Com exceção das filas `*_dev`, é **obrigatório** informar `--time=HH:MM:SS`.  
   - Não usar `--exclusive` nas filas GPU (uso compartilhado do nó).

4. **Conda (wiki 09 – FAQ)**  
   - Criar ambiente em **$SCRATCH** (recomendado): `conda create --prefix $SCRATCH/conda-env python=3.10`.  
   - **Não** deixar o ambiente conda ativado no momento do `sbatch`; carregar e ativar **dentro** do script (`module load anaconda3/...` e `source activate ...`).

5. **Internet nos nós de computação (wiki 03)**  
   - Nós de computação **não** têm acesso irrestrito à internet. Download de pesos/bibliotecas deve ser feito nos **nós de login** (ou solicitar liberação ao Helpdesk).

---

## Checklist antes de submeter

- [ ] Projeto e dados em pasta acessível pelos nós que vão executar o job (recomendado: copiar para `$SCRATCH` e usar `TCC_BASE_DIR=$SCRATCH/tcc-v2-DeeplabV3Plus`).
- [ ] Ambiente conda criado (em `$SCRATCH` ou `$HOME`, conforme acessível nos nós) e **desativado** antes do `sbatch`.
- [ ] `processed_dataset/dataset_registry.yaml` e dados (imagens/máscaras, `pairs.csv`, `splits/`) existem em `TCC_BASE_DIR` (ou no path indicado no registry).
- [ ] Script Slurm: `--time` e `--gpus` definidos; `module load` e `source activate` apenas dentro do script.
- [ ] Submissão feita em um **nó de login** (não no DTN).

---

## Passo a passo recomendado

### 1. Conexão e local do projeto

```bash
# Conferir onde está o projeto (recomendado: SCRATCH para garantir acesso nos nós de execução)
echo $HOME
echo $SCRATCH
```

Se o manual do seu ambiente indicar que Home **não** é acessível nos nós de computação, use SCRATCH:

```bash
cp -r /caminho/local/tcc-v2-DeeplabV3Plus $SCRATCH/
cd $SCRATCH/tcc-v2-DeeplabV3Plus
```

Caso use HOME (e o cluster montar Home nos nós de execução):

```bash
cd $HOME/tcc-v2-DeeplabV3Plus
```

### 2. Ambiente Conda (uma vez; preferir $SCRATCH se os nós não enxergam $HOME)

```bash
module avail anaconda3
module load anaconda3/2024.02_sequana   # ou 2024.10, conforme disponível
# Recomendado: criar em SCRATCH para garantir acesso no job
conda create --prefix $SCRATCH/conda-envs/tcc python=3.10
source activate $SCRATCH/conda-envs/tcc
pip install torch torchvision segmentation-models-pytorch
# instalar o restante do requirements.txt
conda deactivate
```

Se usar ambiente em HOME:

```bash
conda create --prefix $HOME/conda-envs/tcc python=3.10
source activate $HOME/conda-envs/tcc
# ... pip install ...
conda deactivate
```

### 3. Ajustar o script Slurm

Editar `extra/slurm_train_sdumont2nd.slurm` (ou o script que você copiou para o cluster):

- **TCC_BASE_DIR:** deve apontar para a pasta do projeto **no cluster**. Se usou SCRATCH:  
  `export TCC_BASE_DIR=$SCRATCH/tcc-v2-DeeplabV3Plus`
- **Partition:** `-p lncc-h100_shared` (ou a fila GPU do seu projeto; conferir com `sinfo` / wiki 06).
- **module load / source activate:** usar o mesmo `anaconda3` e o mesmo prefix do ambiente que você criou (ex.: `$SCRATCH/conda-envs/tcc` ou `$HOME/conda-envs/tcc`).
- Se for projeto PETROBRAS/Parceiros: descomentar e preencher `#SBATCH --account=SIGLA`.

### 4. Submeter (sem conda ativado)

```bash
cd $SCRATCH/tcc-v2-DeeplabV3Plus   # ou $HOME/...
sbatch extra/slurm_train_sdumont2nd.slurm
# ou, se tiver copiado o script para a raiz:
# sbatch slurm_train_sdumont2nd.slurm
```

### 5. Acompanhar

```bash
squeue -u $USER
squeue --start -u $USER
tail -f slurm_*.out
grep PROGRESS outputs/training/training_*.log
scancel JOBID   # para cancelar
```

Logs: `slurm_JOBID.out`, `slurm_JOBID.err`; métricas e status do treino em `outputs/training/`.

---

## Resumo do que o script Slurm já faz corretamente

- `#SBATCH --time=24:00:00` (obrigatório em filas não-dev).
- `#SBATCH --gpus=1` (obrigatório em fila GPU).
- `module load` e `source activate` **dentro** do script (conda não ativado no sbatch).
- Um nó, uma tarefa, um GPU (`--nodes=1`, `--ntasks-per-node=1`).
- Sem `--exclusive` (compartilhamento de nó).

**Ponto a conferir no seu cluster:** se os nós de execução montam ou não o Home. Em caso de dúvida, use **SCRATCH** para o projeto e para o ambiente conda e defina `TCC_BASE_DIR=$SCRATCH/tcc-v2-DeeplabV3Plus` no script (ou antes do `sbatch`).
