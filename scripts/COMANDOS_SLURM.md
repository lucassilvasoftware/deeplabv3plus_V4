## Ver jobs (o que está rodando / na fila)

| Comando | O que faz |
|---------|-----------|
| `squeue -u $USER` | Seus jobs (pendentes e em execução). |
| `squeue -u $USER -l` | Idem, com mais colunas (inclui **REASON** quando PD). |
| `squeue -p lncc-h100_shared` | Todos os jobs na fila H100. |
| `squeue -O JobID,Partition,StartTime -t PD -u $USER` | Previsão de início (coluna START_TIME) dos seus jobs **pendentes**. |

**Estados (coluna ST):** `PD` = pendente, `R` = rodando, `CD` = concluído, `CA` = cancelado, `F` = falhou, `TO` = estourou o tempo.

---

## Por que o job não sai da fila (REASON)

Quando o job está **PD (PENDING)**, a coluna **REASON** explica:

| REASON | Significado | O que fazer |
|--------|-------------|-------------|
| **QOSMinGRES** | Faltou pedir GPU. | No script: `#SBATCH --gpus=1` (ou --gpus-per-node=1). |
| **QOSMaxGRESPerUser** | Pediu recurso demais ou usou --exclusive. | Nas filas _shared não use --exclusive; não peça mais que metade do nó. |
| **PartitionTimeLimit** | --time maior que o limite da fila. | Reduza --time (ex.: 24:00:00 para lncc-h100_shared). |
| **AssociationJobLimit** | Limite de jobs do projeto/usuário. | Menos jobs ativos ou espere liberar. |
| **Resources** | Aguardando recurso (GPU/nó) ficar livre. | Só esperar. |
| **Dependency** | Job com --dependency aguardando outro. | Normal; esperar o job anterior. |

**Resumo para lncc-h100_shared:** sempre usar `--gpus=1` (ou --gpus-per-node=1) e `--time=HH:MM:SS`; não usar `--exclusive`.

---

## Ver filas e nós (sinfo)

| Comando | O que faz |
|---------|-----------|
| `sinfo` | Resumo de todas as filas e nós. |
| `sinfo -p lncc-h100_shared` | Só a fila H100. |
| `sinfo -p lncc-h100_shared -o "%P %.5a %.10l %.6D %C"` | Partição, disponibilidade, tempo máx., nós, CPUs (alocadas/disponíveis/total). |

---

## Detalhes de um job / histórico

| Comando | O que faz |
|---------|-----------|
| `scontrol show job <JOBID>` | Detalhes do job (ou `scontrol show job <JOBID> -dd` para mais). |
| `sacct -u $USER --format=JobID,JobName,Partition,State,Elapsed,ExitCode` | Histórico dos seus jobs (estado e código de saída). |
| `sacct -j <JOBID> -l` | Detalhes de accounting do job. |

---

## Cancelar job

| Comando | O que faz |
|---------|-----------|
| `scancel <JOBID>` | Cancela um job. |
| `scancel -u $USER` | Cancela todos os seus jobs. |

---

## Onde está a saída / o erro

Depois do `sbatch`, a saída e o erro vão para os arquivos definidos no script (por padrão no mesmo diretório de onde você rodou o sbatch):

- **Saída padrão:** `slurm_<JOBID>.out`
- **Erro padrão:** `slurm_<JOBID>.err`

Exemplo: `sbatch` devolveu `Submitted batch job 12345` → leia `slurm_12345.out` e `slurm_12345.err`.

No nosso projeto, o treino também grava logs em:  
`/scratch/psivgmp/lucas.silva8/deeplabv3plus_V4/outputs/training/training_*.log`

---

## Módulos (ambiente)

| Comando | O que faz |
|---------|-----------|
| `module avail` | Lista módulos disponíveis. |
| `module avail anaconda` | Lista módulos que contêm "anaconda". |
| `module list` | Módulos atualmente carregados. |
| `module load anaconda3/2024.02_sequana` | Carrega um módulo (ex.: Anaconda). |
| `module unload <nome>` | Descarrega um módulo. |

---

## Suas filas e limites (SDumont2nd)

```bash
sacctmgr list user $USER -s format=partition%20,account,MaxJobs,MaxSubmit,MaxWall
```

Mostra partições (filas) que você pode usar, conta (account), máximo de jobs e tempo máximo (MaxWall).

---

## Resumo rápido (copiar/colar)

```bash
# Meus jobs
squeue -u $USER -l

# Só fila H100
squeue -p lncc-h100_shared

# Previsão de início (jobs pendentes)
squeue -O JobID,Partition,StartTime -t PD -u $USER

# Info da fila
sinfo -p lncc-h100_shared

# Cancelar job 12345
scancel 12345

# Ver saída do job 12345
cat slurm_12345.out
cat slurm_12345.err
```

---

**Manuais:**  
- **SDumont2nd (cluster atual):** [manual-sdumont2nd wiki](https://github.com/lncc-sered/manual-sdumont2nd/wiki) — 06 (Gerenciador de filas), 07 (Submeter Jobs), 08 (Verificando status).  
- **SDumont antigo (referência):** [manual-sdumont wiki](https://github.com/lncc-sered/manual-sdumont/wiki) — 03 (Acesso), 04 (Módulos), 06 (Filas), 07 (Submeter), 08 (Status), 09 (FAQ).
