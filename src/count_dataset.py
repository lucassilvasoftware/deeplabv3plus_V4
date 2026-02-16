"""
Script para contar imagens originais e patches gerados por fold.
Execute: python -m src.count_dataset
"""
import cv2
from pathlib import Path
from config import Config
from utils import load_fold_files


def count_patches_per_image(img_path, patch_size=256, stride=128):
    img = cv2.imread(str(img_path))
    if img is None:
        return 0
    H, W = img.shape[:2]
    nh = max(0, (H - patch_size) // stride + 1)
    nw = max(0, (W - patch_size) // stride + 1)
    return nh * nw


def main():
    config = Config()
    print("=" * 50)
    print("Contagem do dataset - imagens vs patches")
    print("=" * 50)

    total_imgs = 0
    total_patches = 0

    for fold_num in range(1, config.N_FOLDS + 1):
        imgs, lbls = load_fold_files(config.DATA_PATH, fold_num)
        n_imgs = len(imgs)
        n_patches = sum(count_patches_per_image(p, config.PATCH_SIZE, config.STRIDE) for p in imgs)

        print(f"\nFold {fold_num}:")
        print(f"  Imagens:  {n_imgs}")
        print(f"  Patches:  {n_patches}")

        total_imgs += n_imgs
        total_patches += n_patches

    # Como cada fold usa imagens diferentes, o total pode ser maior que o dataset real
    # Se os folds forem disjuntos (validação cruzada), total_imgs = dataset inteiro
    print("\n" + "=" * 50)
    print("Totais (somando todos os folds):")
    print(f"  Imagens:  {total_imgs}")
    print(f"  Patches:  {total_patches}")
    if total_imgs > 0:
        print(f"  Média de patches por imagem: {total_patches / total_imgs:.1f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
