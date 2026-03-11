"""
Métricas e helpers para segmentação: IoU/mIoU, F1 por classe e macro, matriz de confusão;
carregamento de folds legado; salvamento de máscara colorida.
Funções principais: compute_iou, compute_f1, compute_confusion_matrix, evaluate_segmentation, print_ious, load_fold_files, save_colored_mask.
"""
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


def format_metrics_table(metrics, class_names, width=18):
    """
    Retorna string formatada para terminal: tabela com IoU, F1, Precision e Recall por classe.
    metrics: dict com 'ious', 'f1_per_class', 'miou', 'f1_macro' e opcionalmente
             'precision_per_class', 'recall_per_class', 'precision_macro', 'recall_macro'.
    """
    def _cell(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "  ---"
        return f"{v*100:5.1f}%"

    ious = metrics["ious"]
    f1s = metrics["f1_per_class"]
    precs = metrics.get("precision_per_class")
    recs = metrics.get("recall_per_class")
    has_prec_rec = precs is not None and recs is not None

    if has_prec_rec:
        lines = []
        lines.append(f"  {'Classe':<{width}} | {'IoU':>7} | {'F1':>7} | {'Prec':>7} | {'Rec':>7}")
        lines.append("  " + "-" * (width + 38))
        for i, name in enumerate(class_names):
            short = (name[: width - 2] + "..") if len(name) > width else name
            p, r = _cell(precs[i]) if i < len(precs) else "  ---", _cell(recs[i]) if i < len(recs) else "  ---"
            lines.append(f"  {short:<{width}} | {_cell(ious[i]):>7} | {_cell(f1s[i]):>7} | {p:>7} | {r:>7}")
        lines.append("  " + "-" * (width + 38))
        prec_mac = metrics.get("precision_macro")
        rec_mac = metrics.get("recall_macro")
        lines.append(f"  {'MÉDIA (mIoU/F1/Prec/Rec)':<{width}} | {_cell(metrics['miou']):>7} | {_cell(metrics['f1_macro']):>7} | {_cell(prec_mac):>7} | {_cell(rec_mac):>7}")
    else:
        lines = []
        lines.append(f"  {'Classe':<{width}} | {'IoU':>7} | {'F1':>7}")
        lines.append("  " + "-" * (width + 18))
        for name, iou, f1 in zip(class_names, ious, f1s):
            short = (name[: width - 2] + "..") if len(name) > width else name
            lines.append(f"  {short:<{width}} | {_cell(iou):>7} | {_cell(f1):>7}")
        lines.append("  " + "-" * (width + 18))
        lines.append(f"  {'MÉDIA (mIoU/F1)':<{width}} | {_cell(metrics['miou']):>7} | {_cell(metrics['f1_macro']):>7}")
    return "\n".join(lines)


def compute_precision_recall(pred, target, num_classes):
    """
    Precision e Recall por classe (binário por classe) e macro.
    Precision = TP/(TP+FP), Recall = TP/(TP+FN).
    Retorna: (precision_per_class, recall_per_class, precision_macro, recall_macro).
    """
    pred = pred.view(-1)
    target = target.view(-1)
    precision_per_class = []
    recall_per_class = []
    for c in range(num_classes):
        tp = ((pred == c) & (target == c)).sum().float().item()
        fp = ((pred == c) & (target != c)).sum().float().item()
        fn = ((pred != c) & (target == c)).sum().float().item()
        prec = float("nan") if (tp + fp) == 0 else (tp / (tp + fp))
        rec = float("nan") if (tp + fn) == 0 else (tp / (tp + fn))
        precision_per_class.append(prec)
        recall_per_class.append(rec)
    valid_prec = [x for x in precision_per_class if not np.isnan(x)]
    valid_rec = [x for x in recall_per_class if not np.isnan(x)]
    precision_macro = np.mean(valid_prec) if valid_prec else 0.0
    recall_macro = np.mean(valid_rec) if valid_rec else 0.0
    return precision_per_class, recall_per_class, precision_macro, recall_macro


def compute_f1(pred, target, num_classes):
    """
    F1 por classe (binário: pred==c vs target==c) e F1 macro.
    pred, target: tensors [N] ou [B,H,W] (serão achatados).
    Retorna: (f1_per_class: list, f1_macro: float).
    """
    pred = pred.view(-1)
    target = target.view(-1)
    f1_per_class = []
    for c in range(num_classes):
        tp = ((pred == c) & (target == c)).sum().float().item()
        fp = ((pred == c) & (target != c)).sum().float().item()
        fn = ((pred != c) & (target == c)).sum().float().item()
        if tp + fp + fn == 0:
            f1_per_class.append(float("nan"))
        else:
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            f1_per_class.append(f1)
    valid = [x for x in f1_per_class if not np.isnan(x)]
    f1_macro = np.mean(valid) if valid else 0.0
    return f1_per_class, f1_macro


def compute_confusion_matrix(pred, target, num_classes):
    """
    Matriz de confusão [num_classes, num_classes]: linhas = ground truth, colunas = predição.
    pred, target: tensors (serão achatados). Retorna np.ndarray int64.
    """
    pred = pred.view(-1).cpu().numpy()
    target = target.view(-1).cpu().numpy()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for i in range(len(pred)):
        t, p = int(target[i]), int(pred[i])
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def evaluate_segmentation(
    pred, target, num_classes, class_names=None, save_dir=None, prefix="eval"
):
    """
    Agrega IoU, mIoU, F1, Precision, Recall e matriz de confusão; opcionalmente salva em save_dir.
    pred, target: tensors (batch de predições e targets).
    Retorna dict com keys: ious, miou, f1_per_class, f1_macro, precision_per_class, recall_per_class,
    precision_macro, recall_macro, confusion_matrix.
    Se save_dir for Path, salva confusion_matrix em CSV e um txt com todas as métricas.
    """
    ious, miou = compute_iou(pred, target, num_classes)
    f1_per_class, f1_macro = compute_f1(pred, target, num_classes)
    precision_per_class, recall_per_class, precision_macro, recall_macro = compute_precision_recall(
        pred, target, num_classes
    )
    cm = compute_confusion_matrix(pred, target, num_classes)
    out = {
        "ious": ious,
        "miou": miou,
        "f1_per_class": f1_per_class,
        "f1_macro": f1_macro,
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "confusion_matrix": cm,
    }
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(save_dir / f"{prefix}_confusion_matrix.csv", cm, fmt="%d", delimiter=",")
        with open(save_dir / f"{prefix}_metrics.txt", "w", encoding="utf-8") as f:
            f.write(f"mIoU: {miou*100:.4f}%\n")
            f.write(f"F1 macro: {f1_macro*100:.4f}%\n")
            f.write(f"Precision macro: {precision_macro*100:.4f}%\n")
            f.write(f"Recall macro: {recall_macro*100:.4f}%\n")
            if class_names:
                f.write("IoU por classe:\n")
                for name, iu in zip(class_names, ious):
                    f.write(f"  {name}: {iu*100:.2f}%\n")
                f.write("F1 por classe:\n")
                for name, f1 in zip(class_names, f1_per_class):
                    f.write(f"  {name}: {f1*100:.2f}%\n")
                f.write("Precision por classe:\n")
                for name, p in zip(class_names, precision_per_class):
                    f.write(f"  {name}: {p*100:.2f}%\n")
                f.write("Recall por classe:\n")
                for name, r in zip(class_names, recall_per_class):
                    f.write(f"  {name}: {r*100:.2f}%\n")
    return out


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
