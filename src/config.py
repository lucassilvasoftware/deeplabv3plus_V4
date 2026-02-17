"""
Configuração central: paths, modelo, dados, treino, loss e early stopping.
Controle de parada só por early stopping (sem scheduler de LR). Equilibrado para boa acurácia sem treino excessivo.

Em cluster (ex.: Santos Dumont): defina TCC_BASE_DIR com o caminho da raiz do projeto no cluster.
Ex.: export TCC_BASE_DIR=/scratch/seudominio/meuprojeto/tcc-v2-DeeplabV3Plus
"""
import os
import torch
from pathlib import Path


def _project_root():
    """Raiz do projeto: TCC_BASE_DIR no cluster, senão pasta que contém src/."""
    env = os.environ.get("TCC_BASE_DIR", "").strip()
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parent.parent


class Config:
    # ----- Paths (única raiz: ROOT; no Santos Dumont defina TCC_BASE_DIR) -----
    ROOT = _project_root()
    PROCESSED_DATASET_ROOT = ROOT / "processed_dataset"
    MODELS_DIR = ROOT / "models"
    # outputs: subpastas para não misturar treino, avaliação e inference
    OUTPUTS_DIR = ROOT / "outputs"
    OUTPUTS_TRAINING = ROOT / "outputs" / "training"   # log, dashboard, training_status.json
    OUTPUTS_EVAL = ROOT / "outputs" / "eval"           # val_best_*, test_apa_*
    OUTPUTS_INFERENCE = ROOT / "outputs" / "inference" # segmented.png, etc.
    DATA_PATH = r"e:\Documents\lulc_env\tcc-fei\data"  # legado (folds)

    # ----- Reprodutibilidade -----
    SEED = 42

    # ----- Modelo -----
    NUM_CLASSES = 8
    ENCODER_NAME = "resnet101"
    ENCODER_WEIGHTS = "imagenet"  # ou None para treinar do zero

    # ----- Dados / DataLoader -----
    BATCH_SIZE = 4  # local/GPU pequena; no Santos Dumont é sobrescrito abaixo
    NUM_WORKERS = 0  # local; no Santos Dumont sobrescrito para 4–8
    APA_MAX_SIZE = (512, 512)  # (1024,1024) no cluster só se aumentar VRAM; ver override abaixo
    # Proporções ao gerar splits automaticamente (ex. dataset 3): train / val / resto=test
    SPLIT_TRAIN_RATIO = 0.70
    SPLIT_VAL_RATIO = 0.15

    # ----- Treino -----
    EPOCHS = 120  # teto máximo; na prática para por early stopping
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2
    GRADIENT_CLIP_NORM = 1.0  # evita explosão de gradiente; 0 = desligado

    # ----- Loss (Combo: Focal + Dice + CE) -----
    LOSS_ALPHA = 0.4  # peso entre Focal+Dice (alpha) e CE (fixo 0.1 no código)
    LOSS_GAMMA = 2.0  # gamma da Focal Loss
    FOCAL_ALPHA = 0.25  # alpha da Focal (peso classe positiva)
    DICE_SMOOTH = 1e-5
    CE_WEIGHT = 0.1  # peso do CrossEntropy no ComboLoss

    # ----- Early stopping (único controle de parada) -----
    EARLY_STOP_PATIENCE = 12  # épocas sem melhorar mIoU na validação → para o treino

    # ----- Modo teste (ativar = treino rápido para validar pipeline) -----
    # TEST_MODE=True  → pesos em models/deeplabv3plus_test_weights.pth (não sobrescreve best; não usar em inference)
    # TEST_MODE=False → pesos em models/deeplabv3plus_best_weights.pth (usar em inference)
    # outputs/ (log, métricas, val_best_*) é sempre preenchido em ambos os modos
    TEST_MODE = True  # True = teste rápido | False = treino completo
    MAX_TRAIN_BATCHES = None  # None = todos; int = limite por época (em TEST_MODE é preenchido automaticamente)
    MAX_VAL_BATCHES = None    # None = todos; int = limite (em TEST_MODE é preenchido automaticamente)

    # ----- Dashboard HTML (atualização em tempo real) -----
    ENABLE_DASHBOARD = True  # False = desligado (ex: cluster/Santos Dumont); True = ativa se DASHBOARD_PORT definido
    DASHBOARD_PORT = 8765  # servidor local; só usado se ENABLE_DASHBOARD=True; None = desligado

    # ----- Device -----
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ----- Classes (APA - ordem Tabela III) -----
    CLASS_NAMES = [
        "Urbano",
        "Veg. Densa",
        "Sombra",
        "Veg. Esparsa",
        "Agricultura",
        "Rocha",
        "Solo Exposto",
        "Água",
    ]
    CLASS_COLORS = {
        0: (128, 128, 128),
        1: (0, 255, 0),
        2: (0, 0, 0),
        3: (173, 255, 47),
        4: (255, 255, 0),
        5: (160, 82, 45),
        6: (160, 82, 45),
        7: (0, 0, 255),
    }

    # ----- Legado (não usado no fluxo processed_dataset) -----
    IMAGE_SIZE = 256
    PATCH_SIZE = 256
    STRIDE = 128
    N_FOLDS = 5

    @classmethod
    def ensure_dirs(cls):
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        cls.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.OUTPUTS_TRAINING.mkdir(parents=True, exist_ok=True)
        cls.OUTPUTS_EVAL.mkdir(parents=True, exist_ok=True)
        cls.OUTPUTS_INFERENCE.mkdir(parents=True, exist_ok=True)


# Em cluster (TCC_BASE_DIR definido): desativa dashboard e ajusta batches para H100/GH200
if os.environ.get("TCC_BASE_DIR"):
    Config.ENABLE_DASHBOARD = False
    Config.BATCH_SIZE = 24          # H100 80GB / GH200: 512x512 cabe bem; reduzir se OOM
    Config.NUM_WORKERS = 6          # Lustre aproveita I/O paralelo
    # Config.APA_MAX_SIZE = (1024, 1024)  # opcional: mais resolução; usar BATCH_SIZE 8–12

# Overrides quando TEST_MODE=True (avalia na importação)
if getattr(Config, "TEST_MODE", False):
    Config.EPOCHS = 8
    Config.EARLY_STOP_PATIENCE = 3
    Config.ENCODER_NAME = "resnet50"
    Config.APA_MAX_SIZE = (512, 512)
    Config.MAX_TRAIN_BATCHES = 30
    Config.MAX_VAL_BATCHES = 30
