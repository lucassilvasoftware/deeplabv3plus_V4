import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


class PetropolisPatchDataset(Dataset):
    def __init__(self, imgs, lbls, config, patch_size=256, stride=128, mode="train"):
        self.imgs, self.lbls = imgs, lbls
        self.config = config
        self.patch_size, self.stride, self.mode = patch_size, stride, mode
        self.patches = []
        self._gen_patches()

        if mode == "train":
            self.tf = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ]
            )
        else:
            self.tf = A.Compose(
                [
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ]
            )

    def _gen_patches(self):
        for img_p, lbl_p in zip(self.imgs, self.lbls):
            img = cv2.cvtColor(cv2.imread(str(img_p)), cv2.COLOR_BGR2RGB)
            mask = self.rgb_to_class(
                cv2.cvtColor(cv2.imread(str(lbl_p)), cv2.COLOR_BGR2RGB)
            )
            H, W = img.shape[:2]
            for i in range(0, H - self.patch_size + 1, self.stride):
                for j in range(0, W - self.patch_size + 1, self.stride):
                    self.patches.append(
                        (
                            img[i : i + self.patch_size, j : j + self.patch_size],
                            mask[i : i + self.patch_size, j : j + self.patch_size],
                        )
                    )

    def rgb_to_class(self, rgb):
        h, w = rgb.shape[:2]
        class_mask = np.zeros((h, w), dtype=np.uint8)
        for cid, color in self.config.CLASS_COLORS.items():
            class_mask[np.all(rgb == color, axis=2)] = cid
        return class_mask

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img, mask = self.patches[idx]
        t = self.tf(image=img, mask=mask)
        return t["image"], t["mask"].long()
