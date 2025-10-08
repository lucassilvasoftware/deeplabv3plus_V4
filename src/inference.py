import torch
from pathlib import Path
from PIL import Image
import numpy as np
import segmentation_models_pytorch as smp
from torchvision import transforms

# ========================
# Configurações
# ========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 8
DATA_DIR = Path(r"E:\Documents\lulc_env\tcc-fei\data\images")
MODEL_PATH = Path(r"E:\Documents\lulc_env\tcc-fei\models\deeplabv3plus_best_fold1_weights.pth")
OUTPUT_DIR = Path(r"E:\Documents\lulc_env\tcc-fei\outputs\ImagensSegmentadas\4 Teste com mudancas")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ========================
# Cores das classes
# ========================
CLASS_COLORS = [
    (0, 100, 0),      # Classe 0
    (0, 255, 0),      # Classe 1
    (128, 128, 128),  # Classe 2
    (160, 82, 45),    # Classe 3
    (255, 255, 0),    # Classe 4
    (0, 0, 255),      # Classe 5
    (255, 0, 0),      # Classe 6
    (0, 0, 0),    # Classe 7
]

# ========================
# Funções
# ========================
def get_model(num_classes):
    model = smp.DeepLabV3Plus(
        encoder_name="resnet101",
        encoder_weights=None,
        classes=num_classes
    )
    return model

def preprocess(img: Image.Image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return transform(img).unsqueeze(0)

def save_mask(mask: np.ndarray, save_path: Path):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, color in enumerate(CLASS_COLORS):
        color_mask[mask == cls_idx] = color
    Image.fromarray(color_mask).save(save_path)

# ========================
# Inferência
# ========================
def main():
    model = get_model(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    for img_path in DATA_DIR.glob("*.*"):
        img = Image.open(img_path).convert("RGB")
        x = preprocess(img).to(DEVICE)
        with torch.no_grad():
            output = model(x)
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        save_mask(pred, OUTPUT_DIR / img_path.name)
        print(f"Imagem segmentada salva: {img_path.name}")

if __name__ == "__main__":
    main()
