"""
Gera dataset_registry.yaml, pairs.csv e splits (train/val/test.txt) a partir de
pastas de imagens e máscaras, sem processar/copiar imagens.
Uso: python generate_dataset_registry.py [--datasets-root PATH] [--out-yaml PATH]
Se --datasets-root não for passado, usa ROOT/datasets_brutos.
Escreve dataset_registry.yaml em processed_dataset/ e, em cada dataset,
pairs.csv e splits/train.txt, val.txt, test.txt.
"""
import argparse
import csv
import random
import sys
from pathlib import Path
from typing import List, Tuple

# raiz do projeto
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

# extensões aceitas para imagem e máscara
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
MASK_EXTENSIONS = {".png"}


def find_pairs(images_dir: Path, masks_dir: Path) -> List[Tuple[str, str]]:
    """Lista pares (nome_imagem, nome_máscara) por stem. Máscara deve ser .png com mesmo stem."""
    if not images_dir.exists() or not masks_dir.exists():
        return []
    stems_to_image = {}
    for f in images_dir.iterdir():
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
            stems_to_image[f.stem] = f.name
    pairs = []
    for f in masks_dir.iterdir():
        if f.is_file() and f.suffix.lower() in MASK_EXTENSIONS:
            stem = f.stem
            if stem in stems_to_image:
                pairs.append((stems_to_image[stem], f.name))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Gera registry + pairs.csv + splits (sem processar imagens)")
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=ROOT / "datasets_brutos",
        help="Raiz onde estão as pastas dos datasets (ex.: 0_lulc_dataset_icmbio_30cm, 1_bizotto_icmbio_3cm)",
    )
    parser.add_argument(
        "--out-yaml",
        type=Path,
        default=ROOT / "processed_dataset" / "dataset_registry.yaml",
        help="Caminho do dataset_registry.yaml a ser escrito",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed para splits")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    datasets_root = args.datasets_root.resolve()
    out_yaml = args.out_yaml.resolve()
    out_yaml.parent.mkdir(parents=True, exist_ok=True)

    # Todos os datasets precisam de splits (train/val/test); use: train = entra no treino; use: test = só avaliação final.
    configs = [
        {
            "key": "lulc_30cm",
            "folder": "0_lulc_dataset_icmbio_30cm",
            "images_dir": "images-tiff",
            "masks_dir": "labels",
            "use": "train",
        },
        {
            "key": "bizotto_3cm",
            "folder": "1_bizotto_icmbio_3cm",
            "images_dir": "images-tiff",
            "masks_dir": "labels",
            "use": "train",
        },
    ]

    registry = {"datasets": {}}
    for cfg in configs:
        dataset_dir = datasets_root / cfg["folder"]
        images_dir = dataset_dir / cfg["images_dir"]
        masks_dir = dataset_dir / cfg["masks_dir"]
        pairs = find_pairs(images_dir, masks_dir)
        if not pairs:
            print(f"[AVISO] Nenhum par em {dataset_dir} (images_dir={cfg['images_dir']}, masks_dir={cfg['masks_dir']})")
            continue
        # path relativo à raiz do projeto (funciona com qualquer --datasets-root)
        try:
            path_str = str(dataset_dir.relative_to(ROOT)).replace("\\", "/")
        except ValueError:
            path_str = str(dataset_dir)
        # Escreve pairs.csv e splits no próprio dataset
        pairs_csv = dataset_dir / "pairs.csv"
        with open(pairs_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image", "mask"])
            for img_name, msk_name in sorted(pairs):
                w.writerow([img_name, msk_name])
        ids = [Path(img).stem for img, _ in sorted(pairs)]
        rng = random.Random(args.seed)
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(n * args.train_ratio)
        n_val = int(n * args.val_ratio)
        n_test = n - n_train - n_val
        train_ids = ids[:n_train]
        val_ids = ids[n_train : n_train + n_val]
        test_ids = ids[n_train + n_val :]
        splits_dir = dataset_dir / "splits"
        splits_dir.mkdir(parents=True, exist_ok=True)
        for name, id_list in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
            with open(splits_dir / f"{name}.txt", "w", encoding="utf-8") as f:
                for i in id_list:
                    f.write(i + "\n")
        print(f"[OK] {cfg['key']}: {len(pairs)} pares, train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")
        registry["datasets"][cfg["key"]] = {
            "path": path_str,
            "use": cfg.get("use", "train"),
            "num_classes": 8,
            "images_dir": cfg["images_dir"],
            "masks_dir": cfg["masks_dir"],
        }

    if not registry["datasets"]:
        print("Nenhum dataset configurado. Verifique --datasets-root e as pastas.")
        sys.exit(1)

    import yaml
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.dump(registry, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"[OK] Registry escrito em {out_yaml}")


if __name__ == "__main__":
    main()
