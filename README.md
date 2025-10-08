# 🌿 LULC-SegNet - Segmentação de Uso e Cobertura do Solo (Petrópolis)

Este projeto implementa uma rede neural de **segmentação semântica** baseada na arquitetura **SegNet com atenção CBAM** para classificar diferentes tipos de uso e cobertura do solo em imagens de satélite do município de Petrópolis.

O modelo é treinado para reconhecer **8 classes** de cobertura do solo:

| Classe | Nome | Cor RGB | Descrição |
|--------|------|---------|-----------|
| 0 | Mata Nativa | (0, 100, 0) | Floresta preservada |
| 1 | Vegetação Densa | (0, 255, 0) | Áreas com densa cobertura vegetal |
| 2 | Ocupação Urbana | (128, 128, 128) | Edificações e infraestrutura urbana |
| 3 | Solo Exposto | (160, 82, 45) | Terrenos sem cobertura vegetal |
| 4 | Corpos d'Água | (0, 0, 255) | Rios, lagos e reservatórios |
| 5 | Agricultura | (255, 255, 0) | Áreas de cultivo e pastagem |
| 6 | Regeneração | (173, 255, 47) | Vegetação em processo de recuperação |
| 7 | Sombra | (0, 0, 0) | Áreas sombreadas |

---

## 📂 Estrutura do Projeto

```
LULC-SegNet/
│
├── src/
│   ├── dataset.py          # Dataset customizado e pré-processamento
│   ├── model.py            # Implementação da LULC-SegNet com CBAM
│   ├── losses.py           # Função de perda (Focal Loss)
│   ├── utils.py            # Funções auxiliares (IoU, carregamento de folds)
│   ├── train.py            # Loop de treinamento e validação
│   ├── main.py             # Script principal para treinamento
│   └── inference.py        # Script para inferência em novas imagens
│
├── data/
│   ├── images/             # Imagens RGB de satélite
│   ├── labels/             # Máscaras coloridas (ground truth)
│   └── folds/              # Divisões para validação cruzada
│       ├── fold1_images.txt
│       ├── fold1_labels.txt
│       ├── fold2_images.txt
│       └── fold2_labels.txt
│
├── models/                 # Modelos treinados salvos
│   └── lulc_segnet_best.pth
│
├── results/                # Resultados de inferência
│   ├── predictions/        # Máscaras preditas
│   └── visualizations/     # Visualizações coloridas
│
├── requirements.txt        # Dependências do projeto
└── README.md              # Esta documentação
```

---

## ⚙️ Pré-requisitos e Instalação

### Requisitos de Sistema
- Python **3.9+**
- CUDA 11.0+ (opcional, mas recomendado para GPU)
- 8GB+ RAM
- 4GB+ espaço em disco

### Instalação

1. **Clone o repositório:**
```bash
git clone https://github.com/seu-usuario/LULC-SegNet.git
cd LULC-SegNet
```

2. **Crie um ambiente virtual:**
```bash
python -m venv lulc_env
source lulc_env/bin/activate  # Linux/Mac
# ou
lulc_env\Scripts\activate     # Windows
```

3. **Instale as dependências:**
```bash
pip install -r requirements.txt
```

### Dependências Principais
```txt
torch>=1.12.0
torchvision>=0.13.0
opencv-python>=4.6.0
numpy>=1.21.0
Pillow>=8.3.0
matplotlib>=3.5.0
tqdm>=4.62.0
scikit-learn>=1.0.0
```

---

## 📊 Preparação dos Dados

### Formato dos Dados
Organize seus dados seguindo esta estrutura:

```
data/
├── images/                 # Imagens RGB (.jpg, .png, .tif)
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── labels/                 # Máscaras coloridas (.png)
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
└── folds/                  # Divisões para validação cruzada
    ├── fold1_images.txt    # Lista de nomes das imagens para treino
    ├── fold1_labels.txt    # Lista de nomes das máscaras para treino
    ├── fold2_images.txt    # Lista de nomes das imagens para validação
    └── fold2_labels.txt    # Lista de nomes das máscaras para validação
```

### Especificações das Máscaras
- **Formato:** PNG com cores RGB exatas
- **Dimensões:** Mesmas das imagens correspondentes
- **Cores:** Devem corresponder exatamente aos valores definidos em `CLASS_COLORS`

### Exemplo de arquivo de fold (fold1_images.txt):
```
image_001.jpg
image_002.jpg
image_003.jpg
```

---

## 🚀 Como Usar

### 1. Treinamento

Para treinar o modelo com os dados preparados:

```bash
cd src/
python main.py
```

**Parâmetros configuráveis em `main.py`:**
```python
class Config:
    BATCH_SIZE = 8
    EPOCHS = 100
    LEARNING_RATE = 0.001
    IMG_SIZE = (256, 256)
    NUM_CLASSES = 8
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
```

### 2. Inferência em Novas Imagens

Para fazer predições em imagens não vistas:

```bash
cd src/
python inference.py --input_path ../data/new_images/ --output_path ../results/ --model_path ../models/lulc_segnet_best.pth
```

**Parâmetros do script de inferência:**
- `--input_path`: Caminho para pasta com imagens de entrada
- `--output_path`: Caminho para salvar os resultados
- `--model_path`: Caminho para o modelo treinado
- `--visualize`: Cria visualizações coloridas (opcional)
- `--batch_size`: Tamanho do batch para inferência (padrão: 4)

### 3. Exemplo de Uso da API

```python
from inference import LULCPredictor

# Inicializar o preditor
predictor = LULCPredictor(model_path='../models/lulc_segnet_best.pth')

# Predizer uma única imagem
mask = predictor.predict_single('path/to/image.jpg')

# Predizer múltiplas imagens
results = predictor.predict_batch(['img1.jpg', 'img2.jpg'])

# Salvar visualização
predictor.save_visualization(mask, 'output/prediction.png')
```

---

## 🧩 Arquitetura do Modelo

### LULC-SegNet com Atenção CBAM

A arquitetura combina:

1. **Encoder-Decoder SegNet:** Estrutura base para segmentação
2. **Blocos Residuais:** Melhor propagação de gradientes
3. **CBAM Attention:** Atenção em canais e espacial
4. **Skip Connections:** Preservação de detalhes de alta resolução

### Componentes Principais

#### Dataset (`dataset.py`)
- Carregamento e pré-processamento de imagens e máscaras
- Augmentações: rotação, flip horizontal, normalização
- Conversão de máscaras coloridas para índices de classe

#### Modelo (`model.py`)
- **Encoder:** 4 blocos de convolução com pooling
- **Attention:** Módulos CBAM entre encoder e decoder
- **Decoder:** 4 blocos de upsampling com concatenação
- **Output:** Camada final com softmax para classificação

#### Função de Perda (`losses.py`)
- **Focal Loss:** Lida com desbalanceamento entre classes
- Foca em exemplos difíceis de classificar
- Reduz contribuição de exemplos fáceis

#### Métricas (`utils.py`)
- **IoU (Intersection over Union):** Por classe e média
- **mIoU:** Métrica principal de avaliação
- **Acurácia por pixel**

---

## 📈 Monitoramento e Resultados

### Durante o Treinamento
O modelo exibe em tempo real:
- **Loss de treino e validação**
- **mIoU por época**
- **IoU individual por classe**
- **Tempo por época**

### Exemplo de Output:
```
Epoch [15/100] - Train Loss: 0.3456 - Val Loss: 0.4123 - mIoU: 0.7234
Class IoUs: [0.82, 0.79, 0.65, 0.71, 0.89, 0.67, 0.74, 0.52]
Best model saved! (mIoU: 0.7234)
```

### Métricas Salvas
- **Modelo com melhor mIoU:** `lulc_segnet_best.pth`
- **Logs de treinamento:** Exibidos no terminal
- **Histórico:** Pode ser salvo modificando `train.py`

---

## 🔧 Customização e Extensões

### Adicionando Novas Classes
1. Modifique `CLASS_COLORS` em `main.py`
2. Ajuste `NUM_CLASSES` na configuração
3. Prepare máscaras com as novas cores
4. Re-treine o modelo

### Modificando Augmentações
Edite a função `get_transforms()` em `dataset.py`:

```python
def get_transforms(is_training=True):
    if is_training:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, p=0.3),
            A.Resize(height=256, width=256),
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
        ])
```

### Ajustando Hiperparâmetros
Principais parâmetros em `main.py`:

```python
class Config:
    BATCH_SIZE = 8          # Ajuste conforme GPU disponível
    EPOCHS = 100           # Número de épocas
    LEARNING_RATE = 0.001  # Taxa de aprendizado
    IMG_SIZE = (256, 256)  # Resolução das imagens
    PATIENCE = 10          # Early stopping
```

---

## 📊 Resultados Esperados

### Performance Típica
- **mIoU geral:** 0.70-0.85 (dependendo da qualidade dos dados)
- **Classes bem classificadas:** Corpos d'água, Mata Nativa
- **Classes desafiadoras:** Sombra, regeneração (devido a similaridades)

### Tempo de Treinamento
- **GPU RTX 3080:** ~30 minutos (100 épocas, 1000 imagens)
- **CPU:** ~3-5 horas (mesmo dataset)

### Uso de Memória
- **GPU:** 4-6GB VRAM (batch_size=8)
- **RAM:** 8-12GB durante treinamento

---

## 🛠️ Solução de Problemas

### Problemas Comuns

**1. Erro de CUDA out of memory:**
```bash
# Reduza o batch_size em main.py
BATCH_SIZE = 4  # ou menor
```

**2. Máscaras com cores incorretas:**
```python
# Verifique se as cores da máscara correspondem exatamente a CLASS_COLORS
# Use um editor de imagem para verificar valores RGB
```

**3. Baixa performance:**
- Verifique qualidade das anotações
- Aumente augmentações
- Ajuste learning rate
- Considere usar pre-trained weights

**4. Modelo não converge:**
- Verifique se as máscaras estão corretas
- Reduza learning rate
- Aumente número de épocas
- Verifique balanceamento das classes

### Debug e Validação

Para verificar se os dados estão corretos:

```python
# Adicione em dataset.py para debug
import matplotlib.pyplot as plt

def visualize_sample(dataset, idx):
    image, mask = dataset[idx]
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(image.permute(1, 2, 0))
    plt.title('Original')
    plt.subplot(132)
    plt.imshow(mask, cmap='tab10')
    plt.title('Ground Truth')
    plt.show()
```

---

## 💡 Próximos Desenvolvimentos

### Funcionalidades Planejadas
- [ ] **Validação cruzada k-fold** automatizada
- [ ] **Visualização interativa** dos resultados
- [ ] **Exportação ONNX/TorchScript** para deploy
- [ ] **API REST** para inferência online
- [ ] **Docker container** para facilitar deployment
- [ ] **Métricas adicionais:** Precisão, Recall, F1-Score por classe
- [ ] **Ensemble de modelos** para melhor performance
- [ ] **Data augmentation avançada** com técnicas específicas para sensoriamento remoto

### Melhorias Técnicas
- [ ] **Mixed precision training** para acelerar treinamento
- [ ] **Learning rate scheduling** adaptativo
- [ ] **TensorBoard logging** para melhor monitoramento
- [ ] **Checkpointing automático** a cada época
- [ ] **Multi-GPU support** para datasets maiores

---

## 📚 Referências e Inspirações

### Papers Relevantes
- **SegNet:** Badrinarayanan, V., et al. "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation."
- **CBAM:** Woo, S., et al. "CBAM: Convolutional Block Attention Module."
- **Focal Loss:** Lin, T.Y., et al. "Focal Loss for Dense Object Detection."

### Datasets Similares
- **ISPRS Potsdam:** Rottensteiner, F., et al.
- **INRIA Aerial Image Dataset**
- **Massachusetts Buildings Dataset**

---

## 🤝 Contribuição

### Como Contribuir
1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

### Diretrizes
- Mantenha o código documentado
- Adicione testes para novas funcionalidades
- Siga as convenções de nomenclatura existentes
- Atualize a documentação quando necessário

---

## 📜 Licença

Este projeto foi desenvolvido como **Trabalho de Conclusão de Curso** para fins acadêmicos. 

**Uso Acadêmico:** Livre para pesquisa e educação  
**Uso Comercial:** Entre em contato com os autores

---

## 👨‍💻 Autores e Contato

**Desenvolvido por:** [Seu Nome]  
**Orientador:** [Nome do Orientador]  
**Instituição:** [Sua Universidade]  

**Contatos:**
- 📧 Email: seu.email@universidade.edu.br
- 📱 LinkedIn: [seu-perfil]
- 🐙 GitHub: [seu-usuario]

---

## 🙏 Agradecimentos

Agradecimentos especiais à:
- **Prefeitura de Petrópolis** pelo fornecimento dos dados
- **Laboratório de Sensoriamento Remoto** pela infraestrutura
- **Comunidade PyTorch** pelas ferramentas excelentes
- **Reviewers e colegas** pelas sugestões valiosas

---

### 📋 Checklist de Implementação

- [x] Arquitetura SegNet com CBAM
- [x] Dataset customizado para LULC
- [x] Focal Loss para classes desbalanceadas
- [x] Script de treinamento completo
- [x] Script de inferência
- [x] Documentação abrangente
- [ ] Validação cruzada k-fold
- [ ] Interface web para demonstração
- [ ] Containerização Docker
- [ ] CI/CD pipeline

---

**⭐ Se este projeto foi útil para você, considere dar uma estrela no GitHub!**

---

*Este README foi criado com ❤️ para facilitar o uso e desenvolvimento do LULC-SegNet.*