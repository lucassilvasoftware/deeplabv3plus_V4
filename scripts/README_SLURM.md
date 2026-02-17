## Informações gerais (wiki)

- **$HOME** e **$SCRATCH** apontam para o Lustre (`/scratch` ou `/petrobr`). Não há HOMEDIR dedicado como no SDumont antigo.
- **Não há backup** dos dados no SDumont2nd; o usuário é responsável por cópias.
- Não é permitido rodar aplicação contínua nos **nós de login** (sdumont[4-7]); processos nesses nós têm limite de **30 minutos**.
- Gerenciador: **Slurm v24.05.3** (ATOS/EVIDEN). Ver suas filas e limites:
  ```bash
  sacctmgr list user $USER -s format=partition%20,account,MaxJobs,MaxSubmit,MaxWall
  ```

---

## Filas com GPU (resumo oficial)

Para **projetos SINAPAD** (ex.: Premium), filas compartilhadas com GPU:

| Fila                 | Wall-clock máx. | Recursos em execução | Jobs submetidos | Arquitetura              |
|----------------------|-----------------|----------------------|-----------------|---------------------------|
| lncc-h100_shared     | 24:00:00        | Equiv. 2 nós         | 30              | Intel + NVIDIA H100      |
| lncc-gh200_shared     | 24:00:00        | Equiv. 2 nós         | 30              | NVIDIA Grace-Hopper GH200|
| lncc-mi300a_shared    | 24:00:00        | Equiv. 1 nó          | 30              | APU AMD MI300A           |

Para **projetos PETROBRAS e Parceiros/ICTs**: filas **ict-h100**, **ict-gh200**, **ict-mi300a**, **petrobr-h100**, etc. Wall-clock e limites diferentes; ver tabela em [06 – Gerenciador de filas](https://github.com/lncc-sered/manual-sdumont2nd/wiki/06-%E2%80%90--Gerenciador-de-filas).

- **Obrigatório** informar **--time=HH:MM:SS**. Sem isso: `Requested time limit is invalid (missing or exceeds some limit)`.
- **Projetos PETROBRAS/Parceiros:** **Obrigatório** informar **--account=sigla-do-projeto**. Sem isso: `Job violates accounting/QOS policy (job submit limit, user's size and/or time limits)`.
- **Filas com GPU:** **Obrigatório** especificar quantidade de GPUs (--gpus, --gpus-per-node, etc.). Sem isso o job fica **pendente com REASON QOSMinGRES**.
- Nas filas **_shared** **não** usar **--exclusive** (job fica com REASON QOSMaxGRESPerUser).

---

## Criar ambiente no cluster (uma vez)

Sim. Você precisa de um ambiente Python no cluster com PyTorch e as dependências do projeto. Faça isso **uma vez** depois de logar (nos nós de login o limite é 30 min; criar env e instalar pacotes costuma caber nesse tempo).

```bash
cd /scratch/psivgmp/lucas.silva8/deeplabv3plus_V4
module load anaconda3/2024.02_sequana
conda create -n tcc python=3.10 -y
source activate tcc
pip install -r requirements.txt
```

Se o módulo Anaconda tiver outro nome no cluster, use `module avail anaconda` (ou `module avail conda`) e ajuste. O nome do env (`tcc`) deve ser o mesmo que está no script SLURM em `source activate tcc`. Se criar com outro nome, edite essa linha no script.

---

## Como rodar o treino

Projeto: FEI (Ciência da Computação) em parceria com professor LNCC — uso das filas SINAPAD (lncc-h100_shared, etc.). **Não** é projeto PETROBRAS/Parceiros; não é necessário `--account`.

### 1. No cluster, entre na pasta do projeto
```bash
cd /scratch/psivgmp/lucas.silva8/deeplabv3plus_V4
```

### 2. Submeta o job
```bash
sbatch scripts/slurm_train_sdumont2nd.slurm
```
O SLURM devolve um **Job ID**. O job entra na fila e roda quando houver recurso.

### 3. Verificar fila e status
```bash
squeue -u $USER                    # seus jobs
squeue -u $USER -l                # mais detalhes
squeue -p lncc-h100_shared        # jobs na fila H100
squeue -O JobID,Partition,StartTime -t PD -u $USER   # previsão de início (jobs pendentes)
```
**Estados (ST):** PD=PENDING, R=RUNNING, CD=COMPLETED, CA=CANCELLED, F=FAILED, TO=TIMEOUT.  
**REASON** (quando PD): QOSMinGRES (falta --gpus), PartitionTimeLimit (--time maior que o da fila), Dependency, etc. Ver [08 – Verificando status](https://github.com/lncc-sered/manual-sdumont2nd/wiki/08-%E2%80%90-Verificando-status-e-utiliza%C3%A7%C3%A3o).

### 4. Saída e erros
- **stdout:** `slurm_<JOBID>.out` (pasta de onde rodou `sbatch`)
- **stderr:** `slurm_<JOBID>.err`
- **Logs do treino (data/hora):** `outputs/training/training_YYYY-MM-DD_HH-MM-SS.log`

### 5. Cancelar job
```bash
scancel <JOBID>
scancel -u $USER   # cancela todos os seus jobs
```

### 6. Histórico (opcional)
```bash
sacct -u $USER --format=JobID,JobName,Partition,State,Elapsed,ExitCode
```

---

## Parâmetros importantes no script SLURM

| Diretiva | Significado |
|----------|-------------|
| `-p lncc-h100_shared` | Fila com GPUs H100 (compartilhada). |
| `--gpus=1` | Uma GPU por job (**obrigatório** em filas GPU). |
| `--time=24:00:00` | Tempo máximo de execução (**obrigatório**). |
| `--job-name=tcc-deeplab` | Nome do job na fila. |
| `--output=slurm_%j.out` | Arquivo de saída; `%j` = Job ID. |
| `--error=slurm_%j.err` | Arquivo de erros. |
| `--account=SIGLA` | Apenas para projetos PETROBRAS/Parceiros (não é o caso FEI/LNCC). |

Não usar `--exclusive` nas filas *_shared.

---

## Fluxo resumido

1. SSH no cluster (VPN + `ssh -o MACs=hmac-sha2-256 -o Ciphers=aes256-ctr lucas.silva8@login.sdumont.lncc.br`).
2. `cd /scratch/psivgmp/lucas.silva8/deeplabv3plus_V4`.
3. `sbatch scripts/slurm_train_sdumont2nd.slurm`.
4. Acompanhe com `squeue -u $USER`; leia `slurm_<JOBID>.out` / `.err` e `outputs/training/training_*.log`.

**Helpdesk:** helpdesk-sdumont@lncc.br — assunto `[SDumont2nd] Erro em submissão/execução de job`.
