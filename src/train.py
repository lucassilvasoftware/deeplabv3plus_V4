"""
Treino e avaliação: fluxo enxuto (sem K-fold). DataLoaders do registry (processed_dataset);
loop train/val; mIoU, F1, Precision, Recall; early stopping; melhor modelo salvo;
avaliação final no test set da APA. Logs via logging.
"""
import logging
import time
import torch
from pathlib import Path
from collections import deque
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import torch.nn.functional as F
from torch import nn
import segmentation_models_pytorch as smp
import numpy as np

from config import Config
from processed_loader import (
    load_registry,
    ensure_splits_for_processed,
    ensure_splits_for_raw,
    build_pair_list,
    ProcessedDataset,
)
from losses import FocalLoss
from utils import (
    evaluate_segmentation,
    format_metrics_table,
)

ROOT = Path(__file__).resolve().parent.parent


def get_model(config):
    encoder_weights = config.ENCODER_WEIGHTS if config.ENCODER_WEIGHTS else None
    return smp.DeepLabV3Plus(
        encoder_name=config.ENCODER_NAME,
        encoder_weights=encoder_weights,
        classes=config.NUM_CLASSES,
    )


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, outputs, targets):
        outputs = torch.softmax(outputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=outputs.shape[1]).permute(0, 3, 1, 2).float()
        intersection = (outputs * targets_one_hot).sum(dim=(2, 3))
        union = outputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class ComboLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, focal_alpha=0.25, dice_smooth=1e-5, ce_weight=0.1):
        super().__init__()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=gamma)
        self.dice = DiceLoss(smooth=dice_smooth)
        self.ce = nn.CrossEntropyLoss()
        self.alpha = alpha
        self.ce_weight = ce_weight

    def forward(self, outputs, targets):
        focal_loss = self.focal(outputs, targets)
        dice_loss = self.dice(outputs, targets)
        ce_loss = self.ce(outputs, targets)
        return self.alpha * focal_loss + (1 - self.alpha) * dice_loss + self.ce_weight * ce_loss


def _build_dataloaders(config):
    """
    Constrói train_loader e val_loader a partir do registry.
    Inclui datasets com use: train. Para datasets com use: test (ex. APA), inclui também
    no treino/val se existir splits/train.txt (e val.txt) — assim você pode usar parte
    da APA no treino para aumentar precisão na APA.
    """
    registry = load_registry(config.PROCESSED_DATASET_ROOT)
    train_datasets = []
    val_datasets = []
    for key, info in registry.items():
        use = info.get("use")
        # path no registry é relativo à raiz do projeto (ROOT) para apontar a datasets brutos ou a subpastas
        dataset_root = config.ROOT / info["path"]
        if not dataset_root.exists():
            continue
        # Datasets de treino: garantir splits
        if use == "train":
            ensure_splits_for_processed(
                dataset_root,
                seed=config.SEED,
                train_ratio=config.SPLIT_TRAIN_RATIO,
                val_ratio=config.SPLIT_VAL_RATIO,
            )
            ensure_splits_for_raw(
                dataset_root,
                info,
                seed=config.SEED,
                train_ratio=config.SPLIT_TRAIN_RATIO,
                val_ratio=config.SPLIT_VAL_RATIO,
            )
        # Dataset de teste (ex. APA): incluir no treino só se tiver train.txt
        elif use == "test":
            if not (dataset_root / "splits" / "train.txt").exists():
                continue
        else:
            continue
        train_pairs = build_pair_list(dataset_root, "train", info=info)
        val_pairs = build_pair_list(dataset_root, "val", info=info)
        num_classes = config.NUM_CLASSES
        # Tamanho alvo único para todos os datasets (evita batch com imagens de tamanhos diferentes)
        target_size = getattr(config, "APA_MAX_SIZE", (1024, 1024))
        if train_pairs:
            train_datasets.append(
                ProcessedDataset(train_pairs, num_classes, mode="train", max_size=target_size)
            )
        if val_pairs:
            val_datasets.append(
                ProcessedDataset(val_pairs, num_classes, mode="val", max_size=target_size)
            )
    if not train_datasets:
        raise RuntimeError("Nenhum dataset de treino encontrado no registry (use: train).")
    train_set = ConcatDataset(train_datasets)
    val_set = ConcatDataset(val_datasets) if val_datasets else None
    train_loader = DataLoader(
        train_set,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE == "cuda"),
    )
    val_loader = (
        DataLoader(
            val_set,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
        )
        if val_set else None
    )
    return train_loader, val_loader


def _set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _log_config(config):
    """Registra hiperparâmetros no início do treino (via logging)."""
    log = logging.getLogger()
    log.info("--- Config ---")
    log.info("SEED=%s BATCH_SIZE=%s EPOCHS=%s", config.SEED, config.BATCH_SIZE, config.EPOCHS)
    log.info("LEARNING_RATE=%s WEIGHT_DECAY=%s", config.LEARNING_RATE, config.WEIGHT_DECAY)
    log.info("GRADIENT_CLIP_NORM=%s", config.GRADIENT_CLIP_NORM)
    log.info("LOSS_ALPHA=%s LOSS_GAMMA=%s CE_WEIGHT=%s", config.LOSS_ALPHA, config.LOSS_GAMMA, config.CE_WEIGHT)
    log.info("EARLY_STOP_PATIENCE=%s", config.EARLY_STOP_PATIENCE)
    log.info("--------------")


def _format_duration(seconds):
    """Formata segundos em HH:MM:SS ou M:SS."""
    if seconds is None or (isinstance(seconds, float) and (np.isnan(seconds) or seconds < 0)):
        return "--:--:--"
    s = int(round(seconds))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def train_model(config):
    config.ensure_dirs()
    _set_seed(config.SEED)
    logging.info("Log de treinamento (splits fixos, sem K-fold)")
    _log_config(config)

    train_loader, val_loader = _build_dataloaders(config)
    n_batches = len(train_loader)
    max_batches = getattr(config, "MAX_TRAIN_BATCHES", None)
    steps_per_epoch = n_batches if max_batches is None else min(n_batches, max_batches)

    # Nome do arquivo de pesos (teste vs real)
    _weights_name = "deeplabv3plus_test_weights.pth" if getattr(config, "TEST_MODE", False) else "deeplabv3plus_best_weights.pth"
    # Header do treino
    logging.info("")
    logging.info("=" * 60)
    logging.info("  TREINAMENTO — DeepLabV3+")
    logging.info("=" * 60)
    logging.info("  Modo: %s", "TESTE (rápido)" if getattr(config, "TEST_MODE", False) else "REAL")
    logging.info("  Pesos: %s", config.MODELS_DIR / _weights_name)
    logging.info("  Épocas: %s  |  Early stop: %s  |  Batch: %s", config.EPOCHS, config.EARLY_STOP_PATIENCE, config.BATCH_SIZE)
    logging.info("  LR: %s  |  Weight decay: %s  |  Clip norm: %s", config.LEARNING_RATE, config.WEIGHT_DECAY, config.GRADIENT_CLIP_NORM)
    logging.info("  Batches/época: %s%s", steps_per_epoch, f" (limitado de {n_batches})" if max_batches else "")
    logging.info("=" * 60)
    logging.info("")

    t_start = time.time()
    epoch_times = []
    batch_times = deque(maxlen=50)
    val_times = deque(maxlen=10)

    last_val_loss = None
    last_miou = None
    last_f1 = None
    last_precision_macro = None
    last_recall_macro = None
    last_ious = []
    last_f1_per_class = []
    last_precision_per_class = []
    last_recall_per_class = []
    miou_history = []  # (epoch, miou) para extrapolação da expectativa final
    model = get_model(config).to(config.DEVICE)
    criterion = ComboLoss(
        alpha=config.LOSS_ALPHA,
        gamma=config.LOSS_GAMMA,
        focal_alpha=config.FOCAL_ALPHA,
        dice_smooth=config.DICE_SMOOTH,
        ce_weight=config.CE_WEIGHT,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scaler = torch.amp.GradScaler("cuda") if config.DEVICE == "cuda" else None
    best_val_miou = 0.0
    patience_counter = 0
    patience = config.EARLY_STOP_PATIENCE

    for epoch in range(1, config.EPOCHS + 1):
        t_epoch_start = time.time()
        model.train()
        train_loss = 0.0
        train_iter = iter(train_loader)
        max_batches = getattr(config, "MAX_TRAIN_BATCHES", None)
        steps = len(train_loader) if max_batches is None else min(len(train_loader), max_batches)
        pbar = tqdm(range(steps), desc=f"Época {epoch}/{config.EPOCHS}", unit="batch")
        for batch_idx in pbar:
            t_batch_start = time.time()
            imgs, masks = next(train_iter)
            imgs, masks = imgs.to(config.DEVICE), masks.to(config.DEVICE)
            optimizer.zero_grad()
            if config.DEVICE == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(imgs)
                    loss = criterion(outputs, masks)
                scaler.scale(loss).backward()
                if config.GRADIENT_CLIP_NORM > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                loss.backward()
                if config.GRADIENT_CLIP_NORM > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP_NORM)
                optimizer.step()
            train_loss += loss.item()
            batch_times.append(time.time() - t_batch_start)
            avg_loss = train_loss / (batch_idx + 1)
            sec_per_batch = np.mean(batch_times) if batch_times else None
            remaining_batches = steps - (batch_idx + 1)
            eta_epoch_sec = (remaining_batches * sec_per_batch) if sec_per_batch else None
            epochs_rem = config.EPOCHS - epoch
            avg_ep = np.mean(epoch_times) if epoch_times else None
            avg_val = np.mean(val_times) if val_times else None
            eta_total_sec = (eta_epoch_sec + (epochs_rem - 1) * avg_ep + epochs_rem * (avg_val or 0)) if (eta_epoch_sec is not None and avg_ep is not None) else (eta_epoch_sec * epochs_rem if eta_epoch_sec and epochs_rem else None)
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg": f"{avg_loss:.4f}",
                "tempo": _format_duration(time.time() - t_start),
                "ETAép": _format_duration(eta_epoch_sec),
                "ETA": _format_duration(eta_total_sec),
                "rest": epochs_rem,
            })

        train_loss /= steps

        # ---------- Resumo da época (train) ----------
        elapsed_so_far = time.time() - t_start
        epochs_remaining = config.EPOCHS - epoch
        avg_ep = np.mean(epoch_times) if epoch_times else None
        avg_val = np.mean(val_times) if val_times else None
        eta_total = (epochs_remaining * (avg_ep + (avg_val or 0))) if (avg_ep is not None and epochs_remaining) else None
        logging.info("")
        logging.info("-" * 60)
        logging.info("  ÉPOCA %s/%s  |  Tempo: %s  |  Épocas rest.: %s  |  ETA: %s",
                    epoch, config.EPOCHS, _format_duration(elapsed_so_far), epochs_remaining, _format_duration(eta_total))
        logging.info("-" * 60)
        logging.info("  Train  loss: %.4f", train_loss)

        if val_loader is not None:
            t_val_start = time.time()
            model.eval()
            preds_all, targets_all, val_loss = [], [], 0.0
            max_val_batches = getattr(config, "MAX_VAL_BATCHES", None)
            val_iter = iter(val_loader)
            val_steps = (len(val_loader) if max_val_batches is None else min(len(val_loader), max_val_batches))
            with torch.no_grad():
                for _ in tqdm(range(val_steps), desc="  Validação", leave=False):
                    imgs, masks = next(val_iter)
                    imgs, masks = imgs.to(config.DEVICE), masks.to(config.DEVICE)
                    if config.DEVICE == "cuda":
                        with torch.amp.autocast(device_type="cuda"):
                            outputs = model(imgs)
                            loss = criterion(outputs, masks)
                    else:
                        outputs = model(imgs)
                        loss = criterion(outputs, masks)
                    val_loss += loss.item()
                    preds_all.append(torch.argmax(outputs, dim=1).cpu())
                    targets_all.append(masks.cpu())
            preds_all = torch.cat(preds_all, dim=0)
            targets_all = torch.cat(targets_all, dim=0)
            val_loss /= val_steps
            val_times.append(time.time() - t_val_start)
            metrics = evaluate_segmentation(
                preds_all, targets_all, config.NUM_CLASSES,
                class_names=config.CLASS_NAMES,
                save_dir=None,
                prefix="val",
            )
            miou = metrics["miou"]
            f1_macro = metrics["f1_macro"]
            last_val_loss = val_loss
            last_miou = miou
            last_f1 = f1_macro
            last_precision_macro = metrics.get("precision_macro")
            last_recall_macro = metrics.get("recall_macro")
            last_ious = metrics["ious"]
            last_f1_per_class = metrics["f1_per_class"]
            last_precision_per_class = metrics.get("precision_per_class", [])
            last_recall_per_class = metrics.get("recall_per_class", [])
            miou_history.append((epoch, miou))

            logging.info("  Val    loss: %.4f", val_loss)
            logging.info("  mIoU: %.2f%%  |  F1 macro: %.2f%%  |  Precision: %.2f%%  |  Recall: %.2f%%",
                        miou * 100, f1_macro * 100,
                        (last_precision_macro or 0) * 100, (last_recall_macro or 0) * 100)
            logging.info("")
            for line in format_metrics_table(metrics, config.CLASS_NAMES).splitlines():
                logging.info("%s", line)

            if miou > best_val_miou:
                best_val_miou = miou
                patience_counter = 0
                torch.save(model.state_dict(), config.MODELS_DIR / _weights_name)
                evaluate_segmentation(
                    preds_all, targets_all, config.NUM_CLASSES,
                    class_names=config.CLASS_NAMES,
                    save_dir=config.OUTPUTS_EVAL,
                    prefix="val_best",
                )
                logging.info("  >> Novo melhor modelo salvo (mIoU %.2f%%)", miou * 100)
            else:
                patience_counter += 1
                logging.info("  Patience: %s/%s (melhor mIoU: %.2f%%)", patience_counter, patience, best_val_miou * 100)
                if patience_counter >= patience:
                    logging.info("  Early stopping ativado.")
                    break
        else:
            torch.save(model.state_dict(), config.MODELS_DIR / _weights_name)

        epoch_times.append(time.time() - t_epoch_start)

        if val_loader is not None:
            logging.info("Época %s: train_loss=%.4f val_loss=%.4f miou=%.2f%% f1=%.2f%% prec=%.2f%% rec=%.2f%%",
                         epoch, train_loss, val_loss, miou * 100, f1_macro * 100,
                         (last_precision_macro or 0) * 100, (last_recall_macro or 0) * 100)
        else:
            logging.info("Época %s: train_loss=%.4f", epoch, train_loss)
        # Uma linha por época para grep no cluster (ex.: grep PROGRESS outputs/training/*.log)
        pct_epoch = 100.0 * epoch / config.EPOCHS
        _vloss = last_val_loss if last_val_loss is not None else 0.0
        _miou = (last_miou * 100) if last_miou is not None else 0.0
        _prec = (last_precision_macro or 0) * 100
        _rec = (last_recall_macro or 0) * 100
        logging.info(
            "PROGRESS epoch=%d/%d pct=%.1f%% train_loss=%.4f val_loss=%.4f miou=%.2f%% f1=%.2f%% prec=%.2f%% rec=%.2f%% best_miou=%.2f%%",
            epoch, config.EPOCHS, pct_epoch, train_loss, _vloss, _miou, (last_f1 or 0) * 100, _prec, _rec, best_val_miou * 100,
        )

    total_elapsed = time.time() - t_start
    logging.info("")
    logging.info("=" * 60)
    logging.info("  TREINO FINALIZADO")
    logging.info("=" * 60)
    logging.info("  Melhor mIoU (val): %.2f%%  |  F1 macro: %.2f%%  |  Precision: %.2f%%  |  Recall: %.2f%%",
                 best_val_miou * 100, (last_f1 or 0) * 100,
                 (last_precision_macro or 0) * 100, (last_recall_macro or 0) * 100)
    logging.info("  Tempo total: %s", _format_duration(total_elapsed))
    logging.info("=" * 60)
    logging.info("")
    return model


def run_final_evaluation_apa(config, model=None, weights_path=None):
    """
    Avaliação final no test set da APA (dataset 0). Carrega modelo de weights_path se model for None.
    Salva métricas e matriz de confusão em outputs/.
    """
    registry = load_registry(config.PROCESSED_DATASET_ROOT)
    apa_key = None
    for key, info in registry.items():
        if info.get("use") == "test":
            apa_key = key
            break
    if apa_key is None:
        logging.info("Nenhum dataset com use: test no registry. Pulando avaliação final APA.")
        return
    info = registry[apa_key]
    dataset_root = config.ROOT / info["path"]
    test_pairs = build_pair_list(dataset_root, "test", info=info)
    if not test_pairs:
        logging.info("Nenhum par no split test da APA. Pulando avaliação final.")
        return
    apa_max = getattr(config, "APA_MAX_SIZE", None)
    test_set = ProcessedDataset(
        test_pairs, config.NUM_CLASSES, mode="val", max_size=apa_max
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )
    config.ensure_dirs()
    if model is None:
        model = get_model(config).to(config.DEVICE)
        path = weights_path or (config.MODELS_DIR / "deeplabv3plus_best_weights.pth")
        if path.exists():
            model.load_state_dict(torch.load(path, map_location=config.DEVICE))
        else:
            logging.warning("Pesos não encontrados em %s. Avaliação com modelo não treinado.", path)
    model.eval()
    preds_all, targets_all = [], []
    with torch.no_grad():
        for imgs, masks in tqdm(test_loader, desc="  Avaliando APA (test)"):
            imgs = imgs.to(config.DEVICE)
            if config.DEVICE == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    out = model(imgs)
            else:
                out = model(imgs)
            preds_all.append(torch.argmax(out, dim=1).cpu())
            targets_all.append(masks)
    preds_all = torch.cat(preds_all, dim=0)
    targets_all = torch.cat(targets_all, dim=0)
    metrics = evaluate_segmentation(
        preds_all, targets_all, config.NUM_CLASSES,
        class_names=config.CLASS_NAMES,
        save_dir=config.OUTPUTS_EVAL,
        prefix="test_apa",
    )
    logging.info("")
    logging.info("=" * 60)
    logging.info("  AVALIAÇÃO FINAL — APA (test set)")
    logging.info("=" * 60)
    logging.info("  mIoU: %.2f%%  |  F1 macro: %.2f%%  |  Precision: %.2f%%  |  Recall: %.2f%%",
                 metrics["miou"] * 100, metrics["f1_macro"] * 100,
                 metrics.get("precision_macro", 0) * 100, metrics.get("recall_macro", 0) * 100)
    logging.info("")
    for line in format_metrics_table(metrics, config.CLASS_NAMES).splitlines():
        logging.info("%s", line)
    logging.info("  Métricas e matriz de confusão: %s (test_apa_*.csv/txt)", config.OUTPUTS_EVAL)
    logging.info("=" * 60)
    logging.info("")
