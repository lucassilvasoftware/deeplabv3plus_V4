# LULC-SegNet — Segmentação de Uso e Cobertura do Solo (Petrópolis)

Segmentação semântica com **SegNet + CBAM** para classificação de uso e cobertura do solo em imagens de satélite. **8 classes:**

| Classe | Nome | RGB | Descrição |
|--------|------|-----|-----------|
| 0 | Mata Nativa | (0, 100, 0) | Floresta preservada |
| 1 | Vegetação Densa | (0, 255, 0) | Cobertura vegetal densa |
| 2 | Ocupação Urbana | (128, 128, 128) | Edificações e infraestrutura |
| 3 | Solo Exposto | (160, 82, 45) | Terreno sem cobertura |
| 4 | Corpos d'Água | (0, 0, 255) | Rios, lagos, reservatórios |
| 5 | Agricultura | (255, 255, 0) | Cultivo e pastagem |
| 6 | Regeneração | (173, 255, 47) | Vegetação em recuperação |
| 7 | Sombra | (0, 0, 0) | Áreas sombreadas |

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

Máscaras: mesmo tamanho das imagens; cores RGB exatas conforme `CLASS_COLORS`. Um nome de arquivo por linha nos `.txt` de folds.

---

## Uso

**Treino:**
```bash
cd src/
python main.py
```

Config em `main.py`: `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE`, `IMG_SIZE`, `NUM_CLASSES`, `DEVICE`.

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

- **Novas classes:** alterar `CLASS_COLORS` e `NUM_CLASSES` em `main.py`; máscaras com as novas cores; retreinar.
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
- **Máscaras erradas:** conferir RGB idêntico a `CLASS_COLORS`.
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
