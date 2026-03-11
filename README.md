# LULC-SegNet — Segmentação de Uso e Cobertura do Solo (Petrópolis)

Segmentação semântica com **DeepLabV3+** (e fluxo legado SegNet + CBAM) para classificação de uso e cobertura do solo em imagens de satélite. **8 classes.**

---

## Classes das máscaras

As máscaras de segmentação (labels) usam **8 classes** com índices de 0 a 7. Cada pixel deve ter **ou** o valor de índice em grayscale (0..7) **ou** a cor RGB exata abaixo. A definição está em `src/config.py` (`CLASS_NAMES` e `CLASS_COLORS`).

| Índice | Nome        | RGB (R, G, B)   | Uso nas máscaras |
|--------|-------------|-----------------|------------------|
| 0      | Urbano      | (128, 128, 128) | Área urbana, edificações |
| 1      | Veg. Densa  | (0, 255, 0)    | Vegetação densa |
| 2      | Sombra      | (0, 0, 0)      | Áreas sombreadas |
| 3      | Veg. Esparsa| (173, 255, 47) | Vegetação esparsa |
| 4      | Agricultura | (255, 255, 0)  | Cultivo, pastagem |
| 5      | Rocha       | (160, 82, 45)  | Rocha |
| 6      | Solo Exposto| (160, 82, 45)  | Solo exposto |
| 7      | Água        | (0, 0, 255)    | Corpos d'água |

**Nota:** As classes 5 (Rocha) e 6 (Solo Exposto) compartilham o mesmo RGB no projeto atual; a distinção é pelo índice. Em máscaras RGB, ambos aparecem com a mesma cor.

Para alterar classes ou cores, edite `CLASS_NAMES` e `CLASS_COLORS` em `src/config.py` e gere/atualize as máscaras de acordo.

---

## Estrutura do Projeto

```
LULC-SegNet/
├── src/
│   ├── dataset.py      # Dataset e pré-processamento
│   ├── model.py        # SegNet + CBAM
│   ├── losses.py       # Focal Loss
│   ├── utils.py        # IoU, folds
│   ├── train.py        # Loop treino/validação
│   ├── main.py         # Entrada de treinamento
│   └── inference.py    # Inferência
├── data/
│   ├── images/         # Imagens RGB
│   ├── labels/         # Máscaras (ground truth)
│   └── folds/          # fold*_images.txt, fold*_labels.txt
├── models/             # Pesos (.pth)
├── results/            # predictions/, visualizations/
├── requirements.txt
└── README.md
```

---

## Requisitos e Instalação

- **Python** 3.9+
- **CUDA** 11.0+ (recomendado)
- **RAM** 8GB+ | **Disco** 4GB+

```bash
git clone https://github.com/seu-usuario/LULC-SegNet.git
cd LULC-SegNet
python -m venv lulc_env
# Linux/Mac: source lulc_env/bin/activate
# Windows:  lulc_env\Scripts\activate
pip install -r requirements.txt
```

**Dependências principais:** `torch`, `torchvision`, `opencv-python`, `numpy`, `Pillow`, `matplotlib`, `tqdm`, `scikit-learn`.

---

## Formato dos Dados

```
data/
├── images/       # .jpg, .png, .tif (RGB)
├── labels/       # .png (máscaras com cores exatas da tabela)
└── folds/
    ├── fold1_images.txt   # Lista de imagens treino
    ├── fold1_labels.txt   # Lista de máscaras treino
    ├── fold2_images.txt   # Validação
    └── fold2_labels.txt   # Validação
```

Máscaras: mesmo tamanho das imagens. Podem ser **grayscale** (valor do pixel = índice 0..7) ou **RGB** com as cores exatas da tabela [Classes das máscaras](#classes-das-máscaras) (`CLASS_COLORS` em `src/config.py`). Um nome de arquivo por linha nos `.txt` de folds.

**Fluxo com `processed_dataset` (recomendado):** O treino lê o registry em `processed_dataset/dataset_registry.yaml`. O `path` de cada dataset é relativo à **raiz do projeto (ROOT)**; assim você pode apontar para pastas de dados brutos (ex.: `datasets_brutos/0_lulc_dataset_icmbio_30cm`) sem copiar arquivos. Use `images_dir` e `masks_dir` no YAML quando os nomes das pastas forem diferentes (ex.: `images-tiff`, `labels`). Para gerar o registry, `pairs.csv` e os splits a partir das pastas de imagens/máscaras (sem processar imagens), rode:
```bash
cd src/
python generate_dataset_registry.py --datasets-root ../datasets_brutos
```
Isso escreve `processed_dataset/dataset_registry.yaml` e, em cada dataset, `pairs.csv` e `splits/train.txt`, `val.txt`, `test.txt`. No cluster (Santos Dumont), defina `TCC_BASE_DIR` com a raiz do projeto; o treino usará `TEST_MODE=False` e hiperparâmetros otimizados (batch 24, LR 2e-4, 8 workers). Para acompanhar o progresso no log: `grep PROGRESS outputs/training/*.log`.

---

## TCC e integração Santos Dumont

**Escolhas metodológicas (para o texto do TCC):**

- **Remoção do K-fold:** O treino usa **splits fixos** (train/val/test) gerados uma vez a partir do registry (proporções 70/15/15%). Isso garante reprodutibilidade e está alinhado a muitos trabalhos de segmentação semântica que reportam métricas em um único split. O código legado com folds permanece em `dataset.py`, `utils.load_fold_files` e `count_dataset.py` apenas para referência.
- **Dataset mesclado:** Dois conjuntos (bizotto 3,7 cm e LULC 30 cm) com as mesmas 8 classes são unidos via `dataset_registry.yaml`; o loader lê `path` relativo à raiz do projeto e aceita `images_dir`/`masks_dir` distintos por fonte. Máscaras podem ser RGB (8 cores) ou grayscale (índice 0..7); a conversão RGB→índice é feita em carga.

**Treino no Santos Dumont (Slurm):**

1. **Preparar o projeto no cluster:** Copie o repositório e os dados para o cluster (ex.: `$HOME/tcc-v2-DeeplabV3Plus` ou `$SCRATCH/...`). Os dados devem estar em `ROOT/datasets_brutos/` (ou no caminho indicado em `path` no registry).
2. **Registry e splits:** Rode uma vez no cluster (ou antes, localmente) o script de geração:  
   `python src/generate_dataset_registry.py --datasets-root ../datasets_brutos`  
   assim `processed_dataset/dataset_registry.yaml` e os `pairs.csv`/splits existem.
3. **Submeter o job:** Nos nós de login do SDumont2nd, defina `TCC_BASE_DIR` (se necessário) e submeta:  
   `sbatch scripts/slurm_train_sdumont2nd.slurm`  
   O script usa partição `lncc-h100_shared`, 1 GPU, 24 h; saída em `slurm_%j.out` e log do treino em `outputs/training/training_*.log`.
4. **Monitorar:** `grep PROGRESS outputs/training/training_*.log` ou `tail -f slurm_*.out`. O job id fica em `outputs/training/slurm_job_id.txt` (útil para `scancel`).

**Hiperparâmetros no cluster (quando `TCC_BASE_DIR` está definido):** `TEST_MODE=False`, `BATCH_SIZE=24`, `LEARNING_RATE=2e-4` (escala com batch maior), `NUM_WORKERS=8`, `EARLY_STOP_PATIENCE=12`, mixed precision (AMP) ativo. Em `src/config.py` é possível descomentar `APA_MAX_SIZE=(1024,1024)` e reduzir o batch (ex.: 8–12) para treinar em resolução maior, se a VRAM permitir.

**Checklist para o treino funcionar:** (a) `processed_dataset/dataset_registry.yaml` existe; (b) cada `path` do registry aponta para uma pasta que existe em `ROOT`; (c) nessas pastas há `pairs.csv` e `splits/` (ou o script de geração foi executado); (d) ambiente Python no cluster com as dependências instaladas; (e) `TCC_BASE_DIR` definido ao rodar (o script Slurm já exporta).

---

## Uso

**Treino:**
```bash
cd src/
python main.py
```

Config em `src/config.py`: `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE`, `NUM_CLASSES`, `APA_MAX_SIZE`, etc.

**Inferência:**
```bash
cd src/
python inference.py --input_path ../data/new_images/ --output_path ../results/ --model_path ../models/lulc_segnet_best.pth
```

Opções: `--visualize`, `--batch_size`.

**API:**
```python
from inference import LULCPredictor
predictor = LULCPredictor(model_path='../models/lulc_segnet_best.pth')
mask = predictor.predict_single('path/to/image.jpg')
predictor.save_visualization(mask, 'output/prediction.png')
```

---

## Arquitetura

- **Encoder-decoder** SegNet + blocos residuais.
- **CBAM** (canal + espacial) entre encoder e decoder.
- **Skip connections** para detalhes de alta resolução.
- **Saída:** softmax por pixel; **loss:** Focal Loss.
- **Métricas:** IoU por classe, mIoU, acurácia por pixel (`utils.py`).

---

## Monitoramento

Durante o treino: loss (treino/validação), mIoU por época, IoU por classe. Melhor modelo salvo como `lulc_segnet_best.pth`.

---

## Customização

- **Novas classes:** alterar `CLASS_COLORS` e `NUM_CLASSES` em `src/config.py`; máscaras com as novas cores; retreinar.
- **Augmentações:** `get_transforms()` em `dataset.py`.
- **Hiperparâmetros:** `main.py` (`BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE`, `IMG_SIZE`, `PATIENCE`).

---

## Performance e Recursos

- **mIoU típico:** 0,70–0,85 (depende dos dados).
- **GPU (ex.: RTX 3080):** ~30 min (100 épocas, ~1k imagens).
- **CPU:** ~3–5 h. **VRAM:** 4–6 GB (batch 8). **RAM:** 8–12 GB.

---

## Problemas Comuns

- **CUDA OOM:** reduzir `BATCH_SIZE` em `main.py`.
- **Máscaras erradas:** conferir RGB idêntico à tabela [Classes das máscaras](#classes-das-máscaras) (`CLASS_COLORS` em `src/config.py`).
- **Baixo mIoU:** checar anotações; aumentar augmentação; ajustar LR ou usar pesos pré-treinados.
- **Não converge:** validar máscaras; diminuir LR; mais épocas; balanceamento de classes.

---

## Referências

- SegNet: Badrinarayanan et al. — encoder-decoder for image segmentation.
- CBAM: Woo et al. — Convolutional Block Attention Module.
- Focal Loss: Lin et al. — dense object detection.

---

## Licença e Contato

Projeto acadêmico (TCC). Uso acadêmico livre; uso comercial: contato com os autores.

**Autor:** [Seu Nome] | **Orientador:** [Nome do Orientador]  
**Email:** seu.email@universidade.edu.br | **GitHub:** [seu-usuario]
