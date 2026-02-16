"""
Verificação rápida antes do treino: registry, datasets, pares e um batch de exemplo.
Rode: python check_before_train.py
Se tudo OK, pode rodar python main.py.
"""
import sys
from pathlib import Path

# raiz do projeto = parent de src
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
from processed_loader import load_registry, ensure_splits_for_processed, build_pair_list, ProcessedDataset
from torch.utils.data import DataLoader


def main():
    cfg = Config()
    print("1. Config e paths")
    print(f"   PROCESSED_DATASET_ROOT = {cfg.PROCESSED_DATASET_ROOT}")
    print(f"   Existe? {cfg.PROCESSED_DATASET_ROOT.exists()}")
    if not cfg.PROCESSED_DATASET_ROOT.exists():
        print("   ERRO: Pasta processed_dataset não encontrada. Coloque-a na raiz do projeto.")
        return
    print()

    print("2. Registry")
    try:
        registry = load_registry(cfg.PROCESSED_DATASET_ROOT)
        for key, info in registry.items():
            use = info.get("use", "?")
            path = info.get("path", "?")
            print(f"   {key}: path={path}, use={use}")
    except Exception as e:
        print(f"   ERRO: {e}")
        return
    print()

    print("3. Splits e pares por dataset")
    for key, info in registry.items():
        dataset_root = cfg.PROCESSED_DATASET_ROOT / info["path"]
        if not dataset_root.exists():
            print(f"   {key}: pasta não existe -> {dataset_root}")
            continue
        ensure_splits_for_processed(dataset_root, seed=cfg.SEED, train_ratio=cfg.SPLIT_TRAIN_RATIO, val_ratio=cfg.SPLIT_VAL_RATIO)
        train_pairs = build_pair_list(dataset_root, "train")
        val_pairs = build_pair_list(dataset_root, "val")
        test_pairs = build_pair_list(dataset_root, "test")
        print(f"   {key}: train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}")
    print()

    print("4. Um batch de exemplo (train)")
    train_datasets = []
    for key, info in registry.items():
        if info.get("use") != "train":
            continue
        dataset_root = cfg.PROCESSED_DATASET_ROOT / info["path"]
        train_pairs = build_pair_list(dataset_root, "train")
        if train_pairs:
            train_datasets.append(ProcessedDataset(train_pairs[:16], cfg.NUM_CLASSES, mode="train", max_size=None))
    if not train_datasets:
        print("   Nenhum dataset de treino. Verifique registry (use: train).")
        return
    from torch.utils.data import ConcatDataset
    ds = ConcatDataset(train_datasets)
    loader = DataLoader(ds, batch_size=min(2, len(ds)), shuffle=False)
    batch_imgs, batch_masks = next(iter(loader))
    print(f"   Batch: imgs shape={batch_imgs.shape}, masks shape={batch_masks.shape}, dtype masks={batch_masks.dtype}")
    print(f"   Valores únicos na máscara: {batch_masks.unique().tolist()}")
    print()

    print("5. APA test (avaliação final)")
    for key, info in registry.items():
        if info.get("use") == "test":
            dataset_root = cfg.PROCESSED_DATASET_ROOT / info["path"]
            test_pairs = build_pair_list(dataset_root, "test")
            print(f"   {key}: {len(test_pairs)} pares no test (avaliação final)")
            break
    else:
        print("   Nenhum dataset com use: test (APA).")

    print()
    print("=== Verificação concluída. Se não houve erro, rode: python main.py ===")


if __name__ == "__main__":
    main()
