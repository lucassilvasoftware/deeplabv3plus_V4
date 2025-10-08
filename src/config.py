import torch


class Config:
    DATA_PATH = r"e:\Documents\lulc_env\tcc-fei\data"
    IMAGE_SIZE = 256
    PATCH_SIZE = 256
    STRIDE = 128
    BATCH_SIZE = 4
    NUM_CLASSES = 8
    CLASS_NAMES = [
        "Mata Nativa",
        "Vegetação Densa",
        "Ocupação Urbana",
        "Solo Exposto",
        "Corpos d'Água",
        "Agricultura",
        "Regeneração",
        "Sombra",
    ]
    CLASS_COLORS = {
        0: (0, 100, 0),
        1: (0, 255, 0),
        2: (128, 128, 128),
        3: (160, 82, 45),
        4: (0, 0, 255),
        5: (255, 255, 0),
        6: (173, 255, 47),
        7: (0, 0, 0),
    }
    EPOCHS = 150
    LEARNING_RATE = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    N_FOLDS = 5
