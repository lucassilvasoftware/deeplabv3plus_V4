import torch
import numpy as np
from pathlib import Path


def compute_iou(pred, target, num_classes):
    ious = []
    pred, target = pred.view(-1), target.view(-1)
    for cls in range(num_classes):
        inter = ((pred == cls) & (target == cls)).sum().float()
        union = ((pred == cls) | (target == cls)).sum().float()
        ious.append(float("nan") if union == 0 else (inter / union).item())
    valid = [x for x in ious if not np.isnan(x)]
    return ious, np.mean(valid) if valid else 0.0


def print_ious(ious, names):
    print("\nIoU por classe:")
    for n, i in zip(names, ious):
        print(f"  {n:15}: {'---' if np.isnan(i) else f'{i*100:6.2f}%'}")


def load_fold_files(data_path, fold_num):
    fold_path = Path(data_path) / "folds"
    imgs_file, lbls_file = fold_path / f"fold{fold_num}_images.txt", fold_path / f"fold{fold_num}_labels.txt"
    if not imgs_file.exists() or not lbls_file.exists():
        raise FileNotFoundError(f"Arquivos do fold {fold_num} não encontrados em {fold_path}")

    with open(imgs_file) as f: imgs = [line.strip() for line in f if line.strip()]
    with open(lbls_file) as f: lbls = [line.strip() for line in f if line.strip()]
    base = Path(data_path)
    return [base / "images" / n for n in imgs], [base / "labels" / n for n in lbls]


def save_colored_mask(mask, class_colors, output_path):
    """
    Converte uma máscara de classes para imagem colorida e salva em disco.
    Args:
        mask (np.ndarray): matriz [H, W] com IDs de classes.
        class_colors (dict): mapeamento {class_id: (R, G, B)}.
        output_path (str ou Path): caminho do arquivo PNG.
    """
    import cv2
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in class_colors.items():
        color_mask[mask == class_id] = color  # RGB

    output_path = Path(output_path).resolve()
    ok = cv2.imwrite(str(output_path), cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))

    if not ok:
        raise RuntimeError(f"Erro ao salvar {output_path}")
    else:
        print(f"Máscara salva em {output_path}")
