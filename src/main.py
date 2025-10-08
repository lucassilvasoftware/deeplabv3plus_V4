from config import Config
from train import train_model
from utils import load_fold_files
import torch

if __name__ == "__main__":

    print("PyTorch version:", torch.__version__)
    print("CUDA available?", torch.cuda.is_available())
    print("Número de GPUs:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("Nome da GPU:", torch.cuda.get_device_name(0))


    fold_num = 1

    train_imgs, train_lbls = load_fold_files(Config.DATA_PATH, fold_num=1)
    print(train_imgs[:5])
    print(train_lbls[:5])

    config = Config()
    print("\n===== Iniciando Treinamento Cross-Validation =====")
    train_model(config)
