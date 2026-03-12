# Filas no Santos Dumont — como verificar e quais usar

Referência: [manual-sdumont wiki — 06 Gerenciador de filas](https://github.com/lncc-sered/manual-sdumont/wiki/06-%E2%80%90--Gerenciador-de-filas).

---

## Como verificar filas disponíveis

### 1. Listar todas as partições (filas) do cluster

```bash
sinfo
```

Saída típica: nome da partição (`PARTITION`), estado (`STATE`: idle, alloc, etc.), tempo limite, nós e CPUs. Para ver só partições GPU:

```bash
sinfo -p sequana_gpu,sequana_gpu_dev1,sequana_gpu_long2,gdl
```

*(No **SDumont2nd** os nomes podem ser outros, por exemplo `lncc-h100_shared`. Use `sinfo` sem filtro para ver a lista real.)*

### 2. Ver **quais filas você tem acesso** e limites

O que importa na prática é o que o **seu usuário/projeto** pode usar:

```bash
sacctmgr list user $USER -s format=partition%20,MaxJobs,MaxSubmit,MaxNodes,MaxCPUs,MaxWall
```

Isso mostra, por fila: nome da partição, máximo de jobs em execução, máximo em fila, máximo de nós, de CPUs e tempo máximo de wall-clock.

### 3. Ver seu projeto e tipo de alocação

Algumas filas ou limites dependem do tipo de projeto (Standard, Educacional, Embaixadores):

```bash
sacctmgr list account $GROUPNAME format=account,descr -P
```

*(Substitua `$GROUPNAME` pelo nome do seu grupo/projeto se souber.)*

### 4. Ver previsão de início de um job pendente

Se o job já estiver na fila:

```bash
squeue --start -u $USER
```

A coluna `START_TIME` indica quando o job deve começar (pode mudar conforme a fila).

---

## Filas GPU (manual — SDumont expansão / 1ª geração)

No manual do SDumont (Bull Sequana X), as filas GPU são:

| Fila                 | Wall-clock máx. | Nós (mín–máx) | Uso típico                          |
|----------------------|-----------------|---------------|-------------------------------------|
| **sequana_gpu**      | 96 h            | 1–21          | Treino longo (até 4 dias)           |
| **sequana_gpu_dev1** | 20 min          | 1–4          | Teste rápido (script, 1 época)       |
| **sequana_gpu_long2**| 744 h (31 dias) | 1–10         | Treino muito longo                  |
| **gdl** (Sequana IA) | 48 h            | 1             | 1 nó com 8× V100 (ML/Deep Learning) |

Regras para **qualquer** fila GPU:

- **Obrigatório** informar `--gpus=N` (ou `--gpus-per-node=N`). Sem isso o job fica pendente com REASON `QOSMinGRES`.
- **Obrigatório** informar `--time=HH:MM:SS`, exceto nas filas `*_dev` (que já têm tempo fixo, ex.: 20 min).
- **Não** usar `--exclusive` (os nós são compartilhados).

---

## Quais filas usar para treino (DeepLab / TCC)

- **Testar o script (subir job, 1–2 épocas):**  
  **sequana_gpu_dev1** — tempo curto (20 min), prioridade maior; ideal para validar comando e ambiente.

- **Treino completo (várias horas / dias):**  
  **sequana_gpu** — até 96 h; 1 GPU já basta para este projeto. Use `--time=24:00:00` (ou o que precisar) e `--gpus=1`.

- **Treino muito longo (semanas):**  
  **sequana_gpu_long2** — até 31 dias; mesma lógica (--gpus, --time).

- **Fila dedicada a IA (quando disponível):**  
  **gdl** — 1 nó, 8 GPUs V100; use se o seu projeto tiver acesso e quiser mais GPUs no mesmo nó.

No **SDumont2nd** (H100, etc.) as filas podem ter outros nomes (ex.: **lncc-h100_shared**). O script deste repositório usa `-p lncc-h100_shared`; confira no seu ambiente com:

```bash
sinfo
sacctmgr list user $USER -s format=partition%20,MaxJobs,MaxSubmit,MaxNodes,MaxCPUs,MaxWall
```

e ajuste o `#SBATCH -p NOME_DA_FILA` no script de submissão.

---

## Resumo rápido

| Objetivo              | Comando principal                                      | Fila sugerida (1ª geração) |
|-----------------------|--------------------------------------------------------|----------------------------|
| Ver todas as filas    | `sinfo`                                               | —                          |
| Ver meu acesso/limites| `sacctmgr list user $USER -s format=partition%20,...` | —                          |
| Teste rápido (job GPU)| 20 min, 1 GPU                                         | sequana_gpu_dev1           |
| Treino completo       | 24–96 h, 1 GPU                                        | sequana_gpu                |
| Treino muito longo    | até 31 dias                                           | sequana_gpu_long2          |

Em caso de dúvida sobre filas ou limites no seu projeto: **helpdesk-sdumont@lncc.br**.
