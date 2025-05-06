import torch
import torch.nn as nn
import torch.nn.functional as F

class MSFP(nn.Module):
    def __init__(self, scales=[1.0, 0.5, 0.25], eps=1e-6):
        super(MSFP, self).__init__()
        self.scales = scales
        self.eps = eps

    def forward(self, x):
        perturbed = []
        for scale in self.scales:
            h, w = int(x.size(2) * scale), int(x.size(3) * scale)
            down = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
            mu = down.mean(dim=[2, 3], keepdim=True)
            std = down.std(dim=[2, 3], keepdim=True) + self.eps
            noise = torch.randn_like(down) * std
            perturbed.append(F.interpolate(down + noise, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False))
        return sum(perturbed) / len(perturbed)