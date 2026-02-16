"""
Carrega o dataset processado (registry YAML + pairs.csv + splits).
Responsável por: leitura do dataset_registry.yaml, resolução de paths por dataset,
construção da lista (image_path, mask_path) por split e Dataset PyTorch unificado.
Funções principais: load_registry, ensure_splits_for_processed, build_pair_list.
Classe: ProcessedDataset.
"""
import csv
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import cv2
import torch
import numpy as np
import yaml
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


def load_registry(processed_root: Path) -> Dict[str, Any]:
    """
    Carrega dataset_registry.yaml da raiz do processed_dataset.
    Retorna dict com chaves = nome do dataset (ex.: icmbio_apa_30cm) e valores = dict
    com path, use, num_classes, gsd_cm (e demais campos do YAML).
    """
    yaml_path = processed_root / "dataset_registry.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Registry não encontrado: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("datasets", {})


def _use_processed_subdir(dataset_root: Path) -> bool:
    """True se o dataset usa pasta processed/ (pairs.csv em processed/)."""
    return (dataset_root / "processed" / "pairs.csv").exists()


def ensure_splits_for_processed(dataset_root: Path, seed: int = 42, train_ratio: float = 0.7, val_ratio: float = 0.15) -> None:
    """
    Se o dataset usa processed/ e não tem splits/train.txt (ex.: dataset 3),
    gera train/val/test.txt a partir de processed/pairs.csv (proporções train_ratio/val_ratio/resto).
    Escreve os arquivos em dataset_root/splits/ e não faz nada se já existirem.
    """
    if not _use_processed_subdir(dataset_root):
        return
    splits_dir = dataset_root / "splits"
    if (splits_dir / "train.txt").exists():
        return
    pairs_path = dataset_root / "processed" / "pairs.csv"
    if not pairs_path.exists():
        return
    ids = []
    with open(pairs_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img = row.get("image", "").strip()
            if not img:
                continue
            base = Path(img).stem
            ids.append(base)
    if not ids:
        return
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    train_ids = ids[:n_train]
    val_ids = ids[n_train : n_train + n_val]
    test_ids = ids[n_train + n_val :]
    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, id_list in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        p = splits_dir / f"{name}.txt"
        with open(p, "w", encoding="utf-8") as f:
            for i in id_list:
                f.write(i + "\n")
    print(f"[processed_loader] Splits criados para {dataset_root.name}: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")


def build_pair_list(
    dataset_root: Path,
    split: str,
    use_processed: Optional[bool] = None,
) -> List[Tuple[Path, Path]]:
    """
    Constrói lista de (path_imagem, path_máscara) para o split dado (train/val/test).
    use_processed: True = pairs em processed/ e IDs no split; False = pairs na raiz, IDs = base name.
    Se use_processed for None, é inferido por _use_processed_subdir.
    Para datasets com processed/ sem splits (ex. dataset 3), chame ensure_splits_for_processed antes.
    """
    if use_processed is None:
        use_processed = _use_processed_subdir(dataset_root)
    splits_dir = dataset_root / "splits"
    split_file = splits_dir / f"{split}.txt"
    if not split_file.exists():
        return []
    with open(split_file, "r", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]

    pairs = []
    if use_processed:
        proc = dataset_root / "processed"
        for bid in ids:
            img_path = proc / f"{bid}.jpg"
            mask_path = proc / f"{bid}_m.png"
            if not img_path.exists():
                img_path = proc / f"{bid}.png"
            if not mask_path.exists():
                continue
            if not img_path.exists():
                continue
            pairs.append((img_path, mask_path))
    else:
        images_dir = dataset_root / "images"
        masks_dir = dataset_root / "masks"
        pairs_csv = dataset_root / "pairs.csv"
        id_set = set(ids)
        if not pairs_csv.exists():
            return []
        with open(pairs_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = row.get("image", "").strip()
                mask_name = row.get("mask", "").strip()
                if not img_name or not mask_name:
                    continue
                base = Path(img_name).stem
                if base not in id_set:
                    continue
                img_path = images_dir / img_name
                mask_path = masks_dir / mask_name
                if img_path.exists() and mask_path.exists():
                    pairs.append((img_path, mask_path))
    return pairs


class ProcessedDataset(Dataset):
    """
    Dataset PyTorch que lê pares (imagem, máscara) de processed_dataset.
    Máscaras já em índices 0..num_classes-1 (PNG modo L). Retorna (image_tensor, mask_long).
    Augmentations apenas para mode='train'.
    """

    def __init__(
        self,
        pairs: List[Tuple[Path, Path]],
        num_classes: int,
        mode: str = "train",
        max_size: Optional[Tuple[int, int]] = None,
    ):
        self.pairs = pairs
        self.num_classes = num_classes
        self.mode = mode
        self.max_size = max_size  # (H, W) opcional para redimensionar imagens grandes (ex. APA)
        if mode == "train":
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ]
            )
        else:
            self.transform = A.Compose(
                [
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ]
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        img_path, mask_path = self.pairs[idx]
        img = cv2.imread(str(img_path))
        if img is None and str(img_path).lower().endswith((".tif", ".tiff")):
            try:
                from PIL import Image
                img = np.array(Image.open(img_path).convert("RGB"))
            except Exception:
                raise RuntimeError(f"Imagem não carregada (tente PIL): {img_path}")
        elif img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            raise RuntimeError(f"Imagem não carregada: {img_path}")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Máscara não carregada: {mask_path}")
        if img.shape[:2] != mask.shape[:2]:
            raise RuntimeError(f"Shape imagem {img.shape[:2]} != máscara {mask.shape[:2]} em {img_path}")
        if self.max_size:
            h, w = self.max_size[0], self.max_size[1]
            if img.shape[0] != h or img.shape[1] != w:
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        transformed = self.transform(image=img, mask=mask)
        image_t = transformed["image"]
        mask_t = transformed["mask"].long()
        mask_t = torch.clamp(mask_t, 0, self.num_classes - 1)
        return image_t, mask_t
