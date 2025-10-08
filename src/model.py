import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class LULCSegNet(nn.Module):
    def __init__(self, num_classes=8, pretrained=True):
        super().__init__()
        resnet = models.resnet34(
            weights=models.ResNet34_Weights.DEFAULT if pretrained else None
        )
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool, self.l1, self.l2, self.l3, self.l4 = (
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.dec4, self.dec3, self.dec2, self.dec1 = (
            nn.ConvTranspose2d(512, 256, 2, 2),
            nn.ConvTranspose2d(256, 128, 2, 2),
            nn.ConvTranspose2d(128, 64, 2, 2),
            nn.ConvTranspose2d(64, 64, 2, 2),
        )
        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.enc1(x), None, None, None, None
        x2 = self.maxpool(x1)
        x2 = self.l1(x2)
        x3 = self.l2(x2)
        x4 = self.l3(x3)
        x5 = self.l4(x4)
        d4, d3, d2, d1 = self.dec4(x5), None, None, None
        d3, d2, d1 = self.dec3(d4), self.dec2(d3), self.dec1(d2)
        out = self.final(d1)
        return F.interpolate(
            out, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False
        )
