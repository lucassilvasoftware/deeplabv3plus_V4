import torch
import numpy as np
import rasterio
from pathlib import Path

from config import Config
from train import get_model   # usa o mesmo modelo do treino
from utils import save_colored_mask

# Diretórios globais
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

config = Config()


# ====== Carregar modelos treinados (ensemble) ======
def load_models():
    models = []
    for fold in range(1, config.N_FOLDS + 1):
        path = MODELS_DIR / f"deeplabv3_best_fold{fold}.pth"
        if not path.exists():
            print(f"Modelo do fold {fold} não encontrado, ignorando.")
            continue
        model = get_model(config.NUM_CLASSES).to(config.DEVICE)
        model.load_state_dict(torch.load(path, map_location=config.DEVICE))
        model.eval()
        models.append(model)
    return models


# ====== Predição com ensemble ======
def predict_ensemble(models, image_tensor):
    with torch.no_grad(), torch.cuda.amp.autocast():
        preds = [torch.softmax(m(image_tensor)["out"], dim=1) for m in models]
        avg_pred = torch.mean(torch.stack(preds), dim=0)
        return torch.argmax(avg_pred, dim=1)


# ====== Previsão em imagem inteira com patches ======
def predict_full_image(models, image_path, patch_size=256):
    with rasterio.open(image_path) as src:
        image = src.read([1, 2, 3])  # bandas RGB
        image = np.transpose(image, (1, 2, 0))  # [H, W, C]

    H, W, _ = image.shape
    segmented = np.zeros((H, W), dtype=np.uint8)

    for y in range(0, H, patch_size):
        for x in range(0, W, patch_size):
            patch = image[y:y+patch_size, x:x+patch_size, :]
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                continue

            patch_tensor = (
                torch.from_numpy(patch)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                .to(config.DEVICE)
                / 255.0
            )
            pred_patch = predict_ensemble(models, patch_tensor)
            segmented[y:y+patch_size, x:x+patch_size] = pred_patch.squeeze().cpu().numpy()

    return segmented


# ====== Relatório de treinamento ======
def training_report():
    log_file = OUTPUTS_DIR / "training_log.txt"
    if not log_file.exists():
        print("\nNenhum log encontrado! Execute o treino primeiro.")
        return

    fold_scores = []
    with open(log_file) as f:
        for line in f:
            if "Melhor mIoU fold" in line:
                score = float(line.strip().split(":")[-1])
                fold_scores.append(score)

    if fold_scores:
        mean, std = np.mean(fold_scores), np.std(fold_scores)
        print("\nRelatório de Treinamento")
        for i, score in enumerate(fold_scores, 1):
            print(f"  Fold {i}: {score:.2f}%")
        print(f"\n  Média: {mean:.2f}%")
        print(f"  Desvio padrão: {std:.2f}%")
    else:
        print("Nenhuma métrica encontrada nos logs.")


# ====== Main ======
if __name__ == "__main__":
    models = load_models()
    if not models:
        print("Nenhum modelo carregado! Treine primeiro.")
        exit()

    test_image = ROOT / "data" / "images" / "raster05.tif"
    print(f"Segmentando {test_image} ...")
    mask = predict_full_image(models, test_image, patch_size=config.IMAGE_SIZE)

    # Salvar resultados
    save_colored_mask(mask, config.CLASS_COLORS, str(OUTPUTS_DIR / "segmented.png"))
    np.save(OUTPUTS_DIR / "segmented_raw.npy", mask)

    print("Segmentação salva em:")
    print(f"  - {OUTPUTS_DIR/'segmented.png'} (colorida)")
    print(f"  - {OUTPUTS_DIR/'segmented_raw.npy'} (matriz classes)")

    training_report()
