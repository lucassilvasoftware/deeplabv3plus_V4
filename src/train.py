import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F
from torch import nn
import segmentation_models_pytorch as smp
from dataset import PetropolisPatchDataset
from losses import FocalLoss
from utils import load_fold_files, compute_iou, print_ious

# ========================
# Diretórios
# ========================
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# ========================
# Modelo: DeepLabV3+ SMP
# ========================
def get_model(num_classes, pretrained_encoder=True):
    encoder_weights = "imagenet" if pretrained_encoder else None
    model = smp.DeepLabV3Plus(
        encoder_name="resnet101",
        encoder_weights=encoder_weights,
        classes=num_classes
    )
    return model

# ========================
# Dice Loss
# ========================
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

# ========================
# Combo Loss: Focal + Dice + CE
# ========================
class ComboLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0):
        super().__init__()
        self.focal = FocalLoss(alpha=0.25, gamma=gamma)
        self.dice = DiceLoss()
        self.ce = nn.CrossEntropyLoss()
        self.alpha = alpha

    def forward(self, outputs, targets):
        focal_loss = self.focal(outputs, targets)
        dice_loss = self.dice(outputs, targets)
        ce_loss = self.ce(outputs, targets)
        return self.alpha * focal_loss + (1 - self.alpha) * dice_loss + 0.1 * ce_loss

# ========================
# Função de treino
# ========================
def train_model(config):
    log_path = OUTPUTS_DIR / "training_log.txt"
    with open(log_path, "w") as f:
        f.write("Log de treinamento\n")

    best_overall_miou = 0.0

    for fold_num in range(1, config.N_FOLDS + 1):
        print(f"\nTreinando Fold {fold_num}/{config.N_FOLDS}")

        train_imgs, train_lbls = load_fold_files(config.DATA_PATH, fold_num)
        val_imgs, val_lbls = load_fold_files(
            config.DATA_PATH, fold_num + 1 if fold_num < config.N_FOLDS else 1
        )

        train_set = PetropolisPatchDataset(train_imgs, train_lbls, config, mode="train")
        val_set = PetropolisPatchDataset(val_imgs, val_lbls, config, mode="val")

        train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE, shuffle=False)

        model = get_model(config.NUM_CLASSES).to(config.DEVICE)
        criterion = ComboLoss(alpha=0.4, gamma=2.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5
        )

        scaler = torch.amp.GradScaler()
        best_fold_miou, patience_counter, patience = 0.0, 0, 15

        for epoch in range(1, config.EPOCHS + 1):
            model.train()
            train_loss = 0
            for imgs, masks in tqdm(train_loader, desc=f"Época {epoch}/{config.EPOCHS}"):
                imgs, masks = imgs.to(config.DEVICE), masks.to(config.DEVICE)
                optimizer.zero_grad()
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(imgs)
                    loss = criterion(outputs, masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()

            # ===== Validação =====
            model.eval()
            preds_all, targets_all, val_loss = [], [], 0
            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs, masks = imgs.to(config.DEVICE), masks.to(config.DEVICE)
                    with torch.cuda.amp.autocast():
                        outputs = model(imgs)
                        loss = criterion(outputs, masks)
                    val_loss += loss.item()
                    preds_all.append(torch.argmax(outputs, dim=1).cpu())
                    targets_all.append(masks.cpu())

            preds_all = torch.cat(preds_all, dim=0)
            targets_all = torch.cat(targets_all, dim=0)
            ious, miou = compute_iou(preds_all, targets_all, config.NUM_CLASSES)

            print(
                f"\nÉpoca {epoch}: Train={train_loss/len(train_loader):.4f}, "
                f"Val={val_loss/len(val_loader):.4f}, mIoU={miou*100:.2f}%"
            )
            print_ious(ious, config.CLASS_NAMES)

            # ===== Salvar apenas state_dict =====
            if miou > best_fold_miou:
                best_fold_miou = miou
                patience_counter = 0
                torch.save(model.state_dict(), MODELS_DIR / f"deeplabv3plus_best_fold{fold_num}_weights.pth")
                print(f"Novo melhor modelo fold {fold_num} salvo! mIoU={miou*100:.2f}%")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping ativado")
                    break

            old_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(miou)
            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr < old_lr:
                print(f"Taxa de aprendizado reduzida: {old_lr:.6f} -> {new_lr:.6f}")

        with open(log_path, "a") as f:
            f.write(f"Melhor mIoU fold {fold_num}: {best_fold_miou*100:.2f}\n")

        best_overall_miou = max(best_overall_miou, best_fold_miou)

    print(f"\nMelhor mIoU geral: {best_overall_miou*100:.2f}%")
