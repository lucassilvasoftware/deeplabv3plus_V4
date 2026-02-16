"""
Treino e avaliação: um único fluxo (sem K-fold). Constrói DataLoaders a partir do registry
(processed_dataset); loop de épocas (train/val); cálculo de mIoU, F1 e matriz de confusão;
early stopping e salvamento do melhor modelo; avaliação final no test set da APA.
Dashboard HTML em tempo real quando DASHBOARD_PORT está definido.
Funções principais: get_model, train_model, run_final_evaluation_apa.
"""
import json
import time
from datetime import datetime, timedelta
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
        dataset_root = config.PROCESSED_DATASET_ROOT / info["path"]
        if not dataset_root.exists():
            continue
        # Datasets de treino: sempre incluir
        if use == "train":
            ensure_splits_for_processed(
                dataset_root,
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
        train_pairs = build_pair_list(dataset_root, "train")
        val_pairs = build_pair_list(dataset_root, "val")
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


def _log_config(config, log_path):
    """Registra hiperparâmetros no início do treino."""
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("--- Config ---\n")
        f.write(f"SEED={config.SEED} BATCH_SIZE={config.BATCH_SIZE} EPOCHS={config.EPOCHS}\n")
        f.write(f"LEARNING_RATE={config.LEARNING_RATE} WEIGHT_DECAY={config.WEIGHT_DECAY}\n")
        f.write(f"GRADIENT_CLIP_NORM={config.GRADIENT_CLIP_NORM}\n")
        f.write(f"LOSS_ALPHA={config.LOSS_ALPHA} LOSS_GAMMA={config.LOSS_GAMMA} CE_WEIGHT={config.CE_WEIGHT}\n")
        f.write(f"EARLY_STOP_PATIENCE={config.EARLY_STOP_PATIENCE}\n")
        f.write("--------------\n")


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


def _json_safe(obj):
    """Converte dict/lista para JSON-safe (nan -> null)."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, (float, np.floating)) and np.isnan(obj):
        return None
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    return obj


def _write_training_status(outputs_dir, state):
    """Escreve training_status.json para o dashboard em tempo real."""
    path = Path(outputs_dir) / "training_status.json"
    safe = _json_safe(state)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(safe, f, indent=2, ensure_ascii=False)


def _start_dashboard_server(outputs_dir, port):
    """Inicia servidor HTTP em thread daemon para servir outputs/ (dashboard HTML + JSON)."""
    import http.server
    import socketserver

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(outputs_dir), **kwargs)

        def log_message(self, format, *args):
            pass  # silencia logs do servidor

        def do_GET(self):
            try:
                super().do_GET()
            except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
                pass  # cliente fechou/atualizou a página durante o envio

    try:
        with socketserver.TCPServer(("", port), Handler) as httpd:
            httpd.serve_forever()
    except OSError:
        pass  # porta em uso ou permissão


def _ensure_dashboard_html(outputs_dir):
    """Escreve training_dashboard.html em outputs_dir (sobrescreve para atualizar template)."""
    path = Path(outputs_dir) / "training_dashboard.html"
    html = _DASHBOARD_HTML
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


# Template do dashboard HTML (atualização em tempo real via fetch do JSON)
_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Treino — DeepLabV3+</title>
  <style>
    * { box-sizing: border-box; }
    body {
      font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;
      margin: 0;
      padding: 2rem;
      background: #f8f9fa;
      color: #1a1a1a;
      font-size: 15px;
      line-height: 1.5;
      max-width: 720px;
      margin-left: auto;
      margin-right: auto;
    }
    h1 {
      font-weight: 600;
      font-size: 1.25rem;
      letter-spacing: -0.02em;
      color: #1a1a1a;
      margin: 0 0 1.5rem 0;
    }
    .card {
      background: #fff;
      border-radius: 10px;
      padding: 1.25rem 1.5rem;
      margin-bottom: 1rem;
      border: 1px solid #e8e8e8;
    }
    .card h2 {
      margin: 0 0 1rem 0;
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: #6b7280;
    }
    .row { display: flex; flex-wrap: wrap; gap: 0.75rem; align-items: center; }
    .metric {
      background: #f4f4f5;
      padding: 0.5rem 0.875rem;
      border-radius: 6px;
      min-width: 6rem;
    }
    .metric span {
      display: block;
      font-size: 0.7rem;
      color: #71717a;
      text-transform: uppercase;
      letter-spacing: 0.03em;
      margin-bottom: 0.15rem;
    }
    .metric strong { font-size: 1rem; font-weight: 600; color: #18181b; }
    .metric-block { margin-top: 0.5rem; min-width: auto; }
    .progress-wrap {
      background: #e4e4e7;
      border-radius: 6px;
      height: 6px;
      overflow: hidden;
      margin: 0.75rem 0;
    }
    .progress-bar {
      background: #18181b;
      height: 100%;
      transition: width 0.25s ease;
    }
    .label-row { margin: 0.5rem 0 0.25rem 0; font-size: 0.8rem; color: #71717a; }
    .label-row strong { color: #18181b; }
    table { width: 100%; border-collapse: collapse; font-size: 0.875rem; }
    th, td { text-align: left; padding: 0.5rem 0.75rem; border-bottom: 1px solid #f4f4f5; }
    th { font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; color: #71717a; }
    td { color: #3f3f46; }
    tr:last-child td { border-bottom: 0; }
    .phase {
      display: inline-block;
      padding: 0.35rem 0.75rem;
      border-radius: 6px;
      font-size: 0.8rem;
      font-weight: 500;
    }
    .phase.training { background: #eff6ff; color: #1d4ed8; }
    .phase.validating { background: #fffbeb; color: #b45309; }
    .phase.finished { background: #f0fdf4; color: #15803d; }
    .updated { color: #a1a1aa; font-size: 0.75rem; margin-top: 1rem; }
  </style>
</head>
<body>
  <h1>DeepLabV3+ · Treino</h1>
  <div class="card">
    <h2>Status</h2>
    <div class="row" style="margin-bottom: 0.5rem;">
      <span class="metric" style="min-width: auto;"><span>Modo</span><strong id="run-mode">--</strong></span>
      <span id="mode-hint" class="label-row" style="margin: 0; font-size: 0.75rem;"></span>
    </div>
    <div id="phase" class="phase">--</div>
    <div class="updated" id="updated">Aguardando dados...</div>
  </div>
    <div class="card">
    <h2>Tempo e progresso</h2>
    <div class="row">
      <div class="metric"><span>Decorrido</span><strong id="elapsed">--:--</strong></div>
      <div class="metric"><span>ETA época</span><strong id="eta-epoch">--:--</strong></div>
      <div class="metric"><span>ETA total</span><strong id="eta-total">--:--</strong></div>
      <div class="metric"><span>Épocas rest.</span><strong id="epochs-left">--</strong></div>
    </div>
    <div class="metric metric-block"><span>Previsão de término</span><strong id="eta-finish">--</strong></div>
    <div class="label-row">Época <strong id="epoch-progress">-- / --</strong></div>
    <div class="progress-wrap"><div class="progress-bar" id="progress-epoch" style="width:0%"></div></div>
    <div class="label-row">Batch <strong id="batch-progress">-- / --</strong></div>
  </div>
  <div class="card">
    <h2>Métricas</h2>
    <div class="row">
      <div class="metric"><span>Train loss</span><strong id="train-loss">--</strong></div>
      <div class="metric"><span>Val loss</span><strong id="val-loss">--</strong></div>
      <div class="metric"><span>mIoU</span><strong id="miou">--%</strong></div>
      <div class="metric"><span>F1 macro</span><strong id="f1">--%</strong></div>
      <div class="metric"><span>Melhor mIoU</span><strong id="best-miou">--%</strong></div>
    </div>
    <div class="metric metric-block"><span>Expectativa precisão final (mIoU)</span><strong id="expected-final-miou">--</strong></div>
  </div>
  <div class="card">
    <h2>IoU / F1 por classe</h2>
    <table><thead><tr><th>Classe</th><th>IoU</th><th>F1</th></tr></thead><tbody id="per-class"></tbody></table>
  </div>
  <script>
    const fmt = (v) => v == null || (typeof v === 'number' && isNaN(v)) ? '--' : v;
    const pct = (v) => v == null || isNaN(v) ? '--' : (v * 100).toFixed(2) + '%';
    function refresh() {
      fetch('training_status.json?t=' + Date.now()).then(r => r.json()).then(d => {
        const testMode = !!d.test_mode;
        document.getElementById('run-mode').textContent = testMode ? 'Teste' : 'Real';
        document.getElementById('run-mode').parentElement.style.background = testMode ? '#fef3c7' : '#d1fae5';
        document.getElementById('mode-hint').textContent = testMode
          ? 'Pesos em models/deeplabv3plus_test_weights.pth (não usar para inference final).'
          : 'Pesos em models/deeplabv3plus_best_weights.pth (uso em inference).';
        document.getElementById('phase').textContent = d.phase || '--';
        document.getElementById('phase').className = 'phase ' + (d.phase || '').toLowerCase();
        document.getElementById('elapsed').textContent = d.elapsed_str || '--:--';
        document.getElementById('eta-epoch').textContent = d.eta_epoch_str || '--:--';
        document.getElementById('eta-total').textContent = d.eta_total_str || '--:--';
        document.getElementById('epochs-left').textContent = fmt(d.epochs_remaining);
        document.getElementById('epoch-progress').textContent = (d.current_epoch ?? '--') + ' / ' + (d.total_epochs ?? '--');
        document.getElementById('progress-epoch').style.width = (d.progress_epoch_pct ?? 0) + '%';
        document.getElementById('batch-progress').textContent = (d.batch_in_epoch ?? '--') + ' / ' + (d.total_batches_epoch ?? '--');
        document.getElementById('eta-finish').textContent = d.eta_finish_at || '--';
        document.getElementById('train-loss').textContent = typeof d.train_loss_epoch === 'number' ? d.train_loss_epoch.toFixed(4) : '--';
        document.getElementById('val-loss').textContent = typeof d.last_val_loss === 'number' ? d.last_val_loss.toFixed(4) : '--';
        document.getElementById('miou').textContent = pct(d.last_val_miou);
        document.getElementById('f1').textContent = pct(d.last_f1_macro);
        document.getElementById('best-miou').textContent = pct(d.best_val_miou);
        document.getElementById('expected-final-miou').textContent = pct(d.expected_final_miou);
        const tbody = document.getElementById('per-class');
        const names = d.class_names || [];
        const ious = d.ious || [];
        const f1s = d.f1_per_class || [];
        tbody.innerHTML = names.map((n, i) => '<tr><td>' + n + '</td><td>' + pct(ious[i]) + '</td><td>' + pct(f1s[i]) + '</td></tr>').join('') || '<tr><td colspan="3">--</td></tr>';
        document.getElementById('updated').textContent = 'Atualizado: ' + new Date().toLocaleTimeString('pt-BR');
      }).catch(() => {});
    }
    refresh();
    setInterval(refresh, 1500);
  </script>
</body>
</html>
"""


def train_model(config):
    config.ensure_dirs()
    MODELS_DIR = config.MODELS_DIR
    OUTPUTS_TRAINING = config.OUTPUTS_TRAINING
    OUTPUTS_EVAL = config.OUTPUTS_EVAL
    log_path = OUTPUTS_TRAINING / "training_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("Log de treinamento (splits fixos, sem K-fold)\n")
    _set_seed(config.SEED)
    _log_config(config, log_path)

    train_loader, val_loader = _build_dataloaders(config)
    n_batches = len(train_loader)
    max_batches = getattr(config, "MAX_TRAIN_BATCHES", None)
    steps_per_epoch = n_batches if max_batches is None else min(n_batches, max_batches)

    # Nome do arquivo de pesos (teste vs real)
    _weights_name = "deeplabv3plus_test_weights.pth" if getattr(config, "TEST_MODE", False) else "deeplabv3plus_best_weights.pth"
    # Header do treino
    print("\n" + "=" * 60)
    print("  TREINAMENTO — DeepLabV3+")
    print("=" * 60)
    print(f"  Modo: {'TESTE (rápido)' if getattr(config, 'TEST_MODE', False) else 'REAL'}")
    print(f"  Pesos: {config.MODELS_DIR / _weights_name}")
    print(f"  Épocas: {config.EPOCHS}  |  Early stop: {config.EARLY_STOP_PATIENCE}  |  Batch: {config.BATCH_SIZE}")
    print(f"  LR: {config.LEARNING_RATE}  |  Weight decay: {config.WEIGHT_DECAY}  |  Clip norm: {config.GRADIENT_CLIP_NORM}")
    print(f"  Batches/época: {steps_per_epoch}" + (f" (limitado de {n_batches})" if max_batches else ""))
    print("=" * 60 + "\n")

    # Dashboard em tempo real (servidor + HTML); desativar com ENABLE_DASHBOARD=False (ex: Santos Dumont)
    dashboard_port = getattr(config, "DASHBOARD_PORT", None) if getattr(config, "ENABLE_DASHBOARD", False) else None
    if dashboard_port:
        _ensure_dashboard_html(OUTPUTS_TRAINING)
        import threading
        server_thread = threading.Thread(
            target=_start_dashboard_server,
            args=(OUTPUTS_TRAINING, dashboard_port),
            daemon=True,
        )
        server_thread.start()
        print(f"  Dashboard (tempo real): http://localhost:{dashboard_port}/training_dashboard.html\n")

    t_start = time.time()
    epoch_times = []  # duração de cada época concluída (train+val)
    batch_times = deque(maxlen=50)
    val_times = deque(maxlen=10)

    def _state(phase, epoch, total_epochs, batch_in_epoch, total_batches_epoch, train_loss_epoch,
               best_miou, last_miou, last_val_loss, last_f1, class_names, ious, f1_per_class, miou_history):
        elapsed = time.time() - t_start
        sec_per_batch = np.mean(batch_times) if batch_times else None
        remaining_batches = total_batches_epoch - batch_in_epoch if batch_in_epoch is not None else 0
        eta_epoch = (remaining_batches * sec_per_batch) if sec_per_batch and remaining_batches > 0 else None
        avg_epoch_time = np.mean(epoch_times) if epoch_times else None
        avg_val_time = np.mean(val_times) if val_times else None
        epochs_rem = (total_epochs - epoch) if (epoch is not None and total_epochs is not None) else None
        if eta_epoch is not None and epochs_rem is not None and avg_epoch_time is not None:
            eta_total = eta_epoch + (epochs_rem - 1) * avg_epoch_time + (epochs_rem * (avg_val_time or 0))
        elif eta_epoch is not None and epochs_rem is not None:
            eta_total = eta_epoch * max(epochs_rem, 1)
        else:
            eta_total = None
        progress_pct = (100.0 * batch_in_epoch / total_batches_epoch) if (batch_in_epoch is not None and total_batches_epoch) else 0
        # Previsão de horário de conclusão
        eta_finish_at = None
        if eta_total is not None and eta_total > 0:
            finish = datetime.now() + timedelta(seconds=eta_total)
            today = datetime.now().date()
            if finish.date() == today:
                eta_finish_at = "Hoje " + finish.strftime("%H:%M")
            elif finish.date() == today + timedelta(days=1):
                eta_finish_at = "Amanhã " + finish.strftime("%H:%M")
            else:
                eta_finish_at = finish.strftime("%d/%m %H:%M")
        # Expectativa de precisão final (regressão linear sobre mIoU por época)
        expected_final_miou = None
        if miou_history and total_epochs and len(miou_history) >= 2:
            x = np.array([e for e, _ in miou_history], dtype=float)
            y = np.array([m for _, m in miou_history], dtype=float)
            n = len(x)
            sx, sy = x.sum(), y.sum()
            sxx = (x * x).sum()
            sxy = (x * y).sum()
            denom = n * sxx - sx * sx
            b = (n * sxy - sx * sy) / denom if denom != 0 else 0
            a = (sy - b * sx) / n
            pred = a + b * total_epochs
            expected_final_miou = float(np.clip(pred, 0, 1))
        out = {
            "test_mode": getattr(config, "TEST_MODE", False),
            "phase": phase,
            "current_epoch": epoch,
            "total_epochs": total_epochs,
            "epochs_remaining": epochs_rem,
            "batch_in_epoch": batch_in_epoch,
            "total_batches_epoch": total_batches_epoch,
            "progress_epoch_pct": round(progress_pct, 1),
            "train_loss_epoch": float(train_loss_epoch) if train_loss_epoch is not None else None,
            "elapsed_seconds": elapsed,
            "elapsed_str": _format_duration(elapsed),
            "eta_epoch_str": _format_duration(eta_epoch),
            "eta_total_str": _format_duration(eta_total),
            "eta_total_seconds": eta_total,
            "eta_finish_at": eta_finish_at,
            "expected_final_miou": expected_final_miou,
            "best_val_miou": float(best_miou) if best_miou is not None else None,
            "last_val_miou": float(last_miou) if last_miou is not None else None,
            "last_val_loss": float(last_val_loss) if last_val_loss is not None else None,
            "last_f1_macro": float(last_f1) if last_f1 is not None else None,
            "class_names": list(class_names) if class_names else [],
            "ious": [float(x) if not np.isnan(x) else None for x in (ious or [])],
            "f1_per_class": [float(x) if not np.isnan(x) else None for x in (f1_per_class or [])],
        }
        return out

    last_val_loss = None
    last_miou = None
    last_f1 = None
    last_ious = []
    last_f1_per_class = []
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
            if dashboard_port:
                st = _state("training", epoch, config.EPOCHS, batch_idx + 1, steps, avg_loss,
                            best_val_miou, last_miou, last_val_loss, last_f1,
                            config.CLASS_NAMES, last_ious, last_f1_per_class, miou_history)
                _write_training_status(OUTPUTS_TRAINING, st)

        train_loss /= steps

        # ---------- Resumo da época (train) ----------
        elapsed_so_far = time.time() - t_start
        epochs_remaining = config.EPOCHS - epoch
        avg_ep = np.mean(epoch_times) if epoch_times else None
        avg_val = np.mean(val_times) if val_times else None
        eta_total = (epochs_remaining * (avg_ep + (avg_val or 0))) if (avg_ep is not None and epochs_remaining) else None
        print("\n" + "-" * 60)
        print(f"  ÉPOCA {epoch}/{config.EPOCHS}  |  Tempo: {_format_duration(elapsed_so_far)}  |  Épocas rest.: {epochs_remaining}  |  ETA: {_format_duration(eta_total)}")
        print("-" * 60)
        print(f"  Train  loss: {train_loss:.4f}")

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
            last_ious = metrics["ious"]
            last_f1_per_class = metrics["f1_per_class"]
            miou_history.append((epoch, miou))
            if dashboard_port:
                st = _state("training", epoch, config.EPOCHS, steps, steps, train_loss,
                            best_val_miou, last_miou, last_val_loss, last_f1,
                            config.CLASS_NAMES, last_ious, last_f1_per_class, miou_history)
                _write_training_status(OUTPUTS_TRAINING, st)

            print(f"  Val    loss: {val_loss:.4f}")
            print(f"  mIoU:  {miou*100:.2f}%   |   F1 macro: {f1_macro*100:.2f}%")
            print()
            print(format_metrics_table(metrics, config.CLASS_NAMES))

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
                print("\n  >> Novo melhor modelo salvo (mIoU {:.2f}%)".format(miou * 100))
            else:
                patience_counter += 1
                print(f"\n  Patience: {patience_counter}/{patience} (melhor mIoU: {best_val_miou*100:.2f}%)")
                if patience_counter >= patience:
                    print("\n  Early stopping ativado.")
                    break
        else:
            torch.save(model.state_dict(), config.MODELS_DIR / _weights_name)

        epoch_times.append(time.time() - t_epoch_start)

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"Época {epoch}: train_loss={train_loss:.4f}")
            if val_loader is not None:
                f.write(f" val_loss={val_loss:.4f} miou={miou*100:.4f} f1_macro={f1_macro*100:.4f}")
            f.write("\n")

    if dashboard_port:
        st = _state("finished", None, config.EPOCHS, None, None, None,
                    best_val_miou, last_miou, last_val_loss, last_f1,
                    config.CLASS_NAMES, last_ious, last_f1_per_class, miou_history)
        st["elapsed_seconds"] = time.time() - t_start
        st["elapsed_str"] = _format_duration(st["elapsed_seconds"])
        _write_training_status(OUTPUTS_TRAINING, st)

    total_elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print("  TREINO FINALIZADO")
    print("=" * 60)
    print(f"  Melhor mIoU (val): {best_val_miou*100:.2f}%")
    print(f"  Tempo total: {_format_duration(total_elapsed)}")
    print("=" * 60 + "\n")
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
        print("Nenhum dataset com use: test no registry. Pulando avaliação final APA.")
        return
    dataset_root = config.PROCESSED_DATASET_ROOT / registry[apa_key]["path"]
    test_pairs = build_pair_list(dataset_root, "test")
    if not test_pairs:
        print("Nenhum par no split test da APA. Pulando avaliação final.")
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
            print(f"Pesos não encontrados em {path}. Avaliação com modelo não treinado.")
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
    print("\n" + "=" * 60)
    print("  AVALIAÇÃO FINAL — APA (test set)")
    print("=" * 60)
    print(f"  mIoU: {metrics['miou']*100:.2f}%   |   F1 macro: {metrics['f1_macro']*100:.2f}%")
    print()
    print(format_metrics_table(metrics, config.CLASS_NAMES))
    print(f"\n  Métricas e matriz de confusão: {config.OUTPUTS_EVAL} (test_apa_*.csv/txt)")
    print("=" * 60 + "\n")
