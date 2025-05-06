# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


#attempt2

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import numpy as np

from lib import wide_resnet
import copy
import scipy.io as io
import random

class DistributionUncertainty(nn.Module):
    """
    Distribution Uncertainty Module
        Args:
        p   (float): probabilty of foward distribution uncertainty module, p in [0,1].
    """

    def __init__(self, p=0.5, eps=1e-6):
        super(DistributionUncertainty, self).__init__()
        self.eps = eps
        self.p = p
        self.factor = 1.0

    def _reparameterize(self, mu, std):
        epsilon = torch.randn_like(std) * self.factor
        return mu + epsilon * std

    def sqrtvar(self, x):
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        t = t.repeat(x.shape[0], 1)
        return t

    def forward(self, x):
        if (not self.training) or (np.random.random()) > self.p:
            return x

        mean = x.mean(dim=[2, 3], keepdim=False)
        std = (x.var(dim=[2, 3], keepdim=False) + self.eps).sqrt()

        sqrtvar_mu = self.sqrtvar(mean)
        sqrtvar_std = self.sqrtvar(std)

        beta = self._reparameterize(mean, sqrtvar_mu)
        gamma = self._reparameterize(std, sqrtvar_std)

        x = (x - mean.reshape(x.shape[0], x.shape[1], 1, 1)) / std.reshape(x.shape[0], x.shape[1], 1, 1)
        x = x * gamma.reshape(x.shape[0], x.shape[1], 1, 1) + beta.reshape(x.shape[0], x.shape[1], 1, 1)
        return x
'''
class LDP(nn.Module):
    """
    Distribution Uncertainty Module
        Args:
        p   (float): probabilty of foward distribution uncertainty module, p in [0,1].
    """

    def __init__(self, hparams, dim, p=0.5, eps=1e-6):
        super(LDP, self).__init__()
        self.eps = eps
        self.p = p
        self.factor = 1.0
        self.hparams = hparams

        self.gamma = nn.Parameter(torch.zeros(dim), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(dim), requires_grad=True)
        self.gamma_margin = nn.Parameter(torch.ones(dim))
        self.beta_margin = nn.Parameter(torch.ones(dim))

        self.count = 0

    def _reparameterize(self, mu, std):
        epsilon = torch.randn_like(std) * self.factor
        return mu + epsilon * std

    def sqrtvar(self, x):
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        t = t.repeat(x.shape[0], 1)
        return t

    def forward(self, x):
        N, C, H, W = x.size()

        mean = x.mean(dim=[2, 3], keepdim=False)
        std = (x.var(dim=[2, 3], keepdim=False) + self.eps).sqrt()

        # Ensure gamma and beta match input channels
        gamma_cloned = self.gamma[:C].clone() if self.gamma.size(0) >= C else self.gamma.clone()
        beta_cloned = self.beta[:C].clone() if self.beta.size(0) >= C else self.beta.clone()

        gamma_cloned = gamma_cloned.view(1, C).expand(N, C)
        beta_cloned = beta_cloned.view(1, C).expand(N, C)

        gamma = std + gamma_cloned
        beta = mean + beta_cloned

        x = (x - mean.reshape(N, C, 1, 1)) / std.reshape(N, C, 1, 1)
        x = x * gamma.contiguous().view(N, C, 1, 1).expand(N, C, H, W) + beta.contiguous().view(N, C, 1, 1).expand(N, C, H, W)
        return x
'''
class LDP(nn.Module):
    def __init__(self, hparams, channels):
        super(LDP, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.alpha = hparams.get('alpha', 1.0)
    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)
        return residual + self.alpha * out

class PermuteAdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, p=0.01, eps=1e-5):
        super(PermuteAdaptiveInstanceNorm2d, self).__init__()
        self.p = p
        self.eps = eps

    def forward(self, x):
        permute = random.random() < self.p
        if permute and self.training:
            perm_indices = torch.randperm(x.size()[0])
        else:
            return x
        size = x.size()
        N, C, H, W = size
        if (H, W) == (1, 1):
            print('encountered bad dims')
            return x
        return adaptive_instance_normalization(x, x[perm_indices], self.eps)

    def extra_repr(self) -> str:
        return 'p={}'.format(
            self.p
        )
        
def adaptive_instance_normalization(content_feat, style_feat, eps=1e-5):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat.detach(), eps)
    content_mean, content_std = calc_mean_std(content_feat, eps)
    content_std = content_std + eps  # to avoid division by 0
    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert (len(size) == 4)
    N, C, H, W = size
    feat_std = torch.sqrt(feat.view(N, C, -1).var(dim=2).view(N, C, 1, 1) + eps)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def remove_batch_norm_from_resnet(model):
    fuse = torch.nn.utils.fusion.fuse_conv_bn_eval
    model.eval()

    model.conv1 = fuse(model.conv1, model.bn1)
    model.bn1 = Identity()

    for name, module in model.named_modules():
        if name.startswith("layer") and len(name) == 6:
            for b, bottleneck in enumerate(module):
                for name2, module2 in bottleneck.named_modules():
                    if name2.startswith("conv"):
                        bn_name = "bn" + name2[-1]
                        setattr(bottleneck, name2,
                                fuse(module2, getattr(bottleneck, bn_name)))
                        setattr(bottleneck, bn_name, Identity())
                if isinstance(bottleneck.downsample, torch.nn.Sequential):
                    bottleneck.downsample[0] = fuse(bottleneck.downsample[0],
                                                    bottleneck.downsample[1])
                    bottleneck.downsample[1] = Identity()
    model.train()
    return model
'''
class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
'''

class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([
            nn.Linear(hparams['mlp_width'], hparams['mlp_width'])
            for _ in range(hparams['mlp_depth']-2)])
        self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x

class ResNet(torch.nn.Module):
    def __init__(self, input_shape, hparams, network=None):
        super(ResNet, self).__init__()
        if network is None:
            if hparams.get('resnet18', False):
                self.network = torchvision.models.resnet18(weights='IMAGENET1K_V1')
                self.n_outputs = 512
            else:
                self.network = torchvision.models.resnet50(weights='IMAGENET1K_V1')
                self.n_outputs = 2048
        else:
            self.network = network
        self.dropout = nn.Dropout(hparams.get('resnet_dropout', 0.0))
        self.hparams = hparams
        self.norm2_conv1 = LDP(hparams, 64)
        self.norm2_maxpool = LDP(hparams, 64)
        self.norm2_layer1 = LDP(hparams, 256)
        self.norm2_layer2 = LDP(hparams, 512)
        self.norm2_layer3 = LDP(hparams, 1024)
        self.norm2_layer4 = LDP(hparams, 2048)
        self.x0 = self.x1 = self.x2 = self.x3 = self.x4 = None
    

    def forward(self, x, perturb=False):
        # Shallow features (after layer1)
        x = self.network.conv1(x)
        x = self.network.bn1(x)
        x = self.network.relu(x)
        x = self.network.maxpool(x)
        shallow_feat = self.network.layer1(x)
        
        # Mid features (after layer3)
        x = self.network.layer2(shallow_feat)
        mid_feat = self.network.layer3(x)
        
        # Deep features (after layer4, before avgpool)
        deep_feat = self.network.layer4(mid_feat)
        
        return shallow_feat, mid_feat, deep_feat

    def train(self, mode=True):
        
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

class ResNet_tea(torch.nn.Module):
    def __init__(self, input_shape, hparams):
        super(ResNet_tea, self).__init__()
        if hparams.get('resnet18', False):
            self.network = torchvision.models.resnet18(weights='IMAGENET1K_V1')
            self.n_outputs = 512
        else:
            self.network = torchvision.models.resnet50(weights='IMAGENET1K_V1')
            self.n_outputs = 2048
        self.dropout = nn.Dropout(hparams.get('resnet_dropout', 0.0))
        self.hparams = hparams
        self.norm2_conv1 = LDP(hparams, 64)
        self.norm2_maxpool = LDP(hparams, 64)
        self.norm2_layer1 = LDP(hparams, 256)
        self.norm2_layer2 = LDP(hparams, 512)
        self.norm2_layer3 = LDP(hparams, 1024)
        self.norm2_layer4 = LDP(hparams, 2048)
        self.x0 = self.x1 = self.x2 = self.x3 = self.x4 = None

    '''
    def forward(self, x, perturb=False):
        
        
        x = self.network.conv1(x)
        if perturb:
            x = self.norm2_conv1(x)
        self.x0 = x
        x = self.network.bn1(x)
        x = self.network.relu(x)
        x = self.network.maxpool(x)
        if perturb:
            x = self.norm2_maxpool(x)
        self.x1 = x
        x = self.network.layer1(x)
        if perturb:
            x = self.norm2_layer1(x)
        self.x2 = x
        x = self.network.layer2(x)
        if perturb:
            x = self.norm2_layer2(x)
        self.x3 = x
        x = self.network.layer3(x)
        if perturb:
            x = self.norm2_layer3(x)
        self.x4 = x
        x = self.network.layer4(x)
        if perturb:
            x = self.norm2_layer4(x)
        x = self.network.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.network.fc(x)
        output = self.dropout(x)
        return output
    '''

    def forward(self, x, perturb=False):
        # Apply perturbations at specified layers
        x = self.network.conv1(x)
        if perturb:
            x = self.norm2_conv1(x)
        self.x0 = x
        x = self.network.bn1(x)
        x = self.network.relu(x)
        x = self.network.maxpool(x)
        if perturb:
            x = self.norm2_maxpool(x)
        self.x1 = x
        shallow_feat = self.network.layer1(x)
        if perturb:
            shallow_feat = self.norm2_layer1(shallow_feat)
        self.x2 = shallow_feat
        
        # Mid features (after layer3)
        x = self.network.layer2(shallow_feat)
        if perturb:
            x = self.norm2_layer2(x)
        self.x3 = x
        mid_feat = self.network.layer3(x)
        if perturb:
            mid_feat = self.norm2_layer3(mid_feat)
        self.x4 = mid_feat
        
        # Deep features (after layer4, before avgpool)
        deep_feat = self.network.layer4(mid_feat)
        if perturb:
            deep_feat = self.norm2_layer4(deep_feat)
        
        return shallow_feat, mid_feat, deep_feat

    def train(self, mode=True):
        
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

class MNIST_CNN(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """
    n_outputs = 128

    def __init__(self, input_shape, hparams):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)
        self.x0 = x

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)
        self.x1 = x

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)
        self.x2 = x

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.avgpool(x)
        x = x.view(len(x), -1)
        return x

class MNIST_CNN_tea(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """
    n_outputs = 128

    def __init__(self, input_shape, hparams):
        super(MNIST_CNN_tea, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.norm0 = LDP(hparams, 64)
        self.norm1 = LDP(hparams, 128)
        self.norm2 = LDP(hparams, 128)
        self.norm3 = LDP(hparams, 128)
        self.norm4 = LDP(hparams, 1024)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x, perturb=False):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)
        if perturb:
            x = self.norm0(x)
        self.x0 = x

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)
        if perturb:
            x = self.norm1(x)
        self.x1 = x

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)
        if perturb:
            x = self.norm2(x)
        self.x2 = x

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)
        self.x3 = x

        x = self.avgpool(x)
        x = x.view(len(x), -1)
        return x

class ContextNet(nn.Module):
    def __init__(self, input_shape):
        super(ContextNet, self).__init__()

        # Keep same dimensions
        padding = (5 - 1) // 2
        self.context_net = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 5, padding=padding),
        )

    def forward(self, x):
        return self.context_net(x)
'''
def Featurizer(input_shape, hparams, is_norm=0):
    """Auto-select an appropriate featurizer for the given input shape."""
    if len(input_shape) == 1:
        return MLP(input_shape[0], hparams["mlp_width"], hparams)
    elif input_shape[1:3] == (28, 28):
        if not is_norm:
            return MNIST_CNN(input_shape, hparams)
        else:
            return MNIST_CNN_tea(input_shape, hparams)
    elif input_shape[1:3] == (32, 32):
        return wide_resnet.Wide_ResNet(input_shape, 16, 2, 0.)
    elif input_shape[1:3] == (224, 224):
        if not is_norm:
            return ResNet(input_shape, hparams)
        else:
            return ResNet_tea(input_shape, hparams)
    else:
        raise NotImplementedError
'''

class Featurizer(nn.Module):
    def __init__(self, input_shape, hparams, is_norm=0):
        super(Featurizer, self).__init__()
        if input_shape[0] != 3:
            raise ValueError(f"Expected 3 channels, got {input_shape[0]}")
        if input_shape[1:3] == (224, 224):
            if not is_norm:
                self.network = ResNet(input_shape, hparams)
            else:
                self.network = ResNet_tea(input_shape, hparams)
        else:
            raise ValueError(f"Unsupported input shape: {input_shape}")
        self.n_outputs = self.network.n_outputs
        self.x0 = self.x1 = self.x2 = self.x3 = self.x4 = None

    def forward(self, x, perturb=False):
        output = self.network(x, perturb=perturb)
        self.x0 = self.network.x0
        self.x1 = self.network.x1
        self.x2 = self.network.x2
        self.x3 = self.network.x3
        self.x4 = self.network.x4
        return output

'''
class Classifier(nn.Module):
    def __init__(self, n_inputs, n_classes, nonlinear=True):
        super(Classifier, self).__init__()
        if nonlinear:
            self.fc = nn.Sequential(
                nn.Linear(n_inputs, 256),
                nn.ReLU(),
                nn.Linear(256, n_classes)
            )
        else:
            self.fc = nn.Linear(n_inputs, n_classes)

    def forward(self, x):
        return self.fc(x)
'''
'''
class Classifier(nn.Module):
    def __init__(self, in_features, out_features, hparams=None):
        super(Classifier, self).__init__()
        # Shallow feature classifier
        self.shallow_fc = nn.Linear(256, out_features)  # Adjust 256 to your shallow feature dim
        
        # Mid feature classifier
        self.mid_fc = nn.Linear(512, out_features)    # Adjust 512 to your mid feature dim
        
        # Deep feature classifier
        self.deep_fc = nn.Linear(2048, out_features)   # For ResNet50 deep features
        
        # Weighting parameters
        self.alpha = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))
        
    def forward(self, shallow_feat, mid_feat, deep_feat):
        shallow_logits = self.shallow_fc(shallow_feat)
        mid_logits = self.mid_fc(mid_feat)
        deep_logits = self.deep_fc(deep_feat)
        
        # Weighted combination
        total_logits = (self.alpha[0] * shallow_logits + 
                       self.alpha[1] * mid_logits + 
                       self.alpha[2] * deep_logits)
        return total_logits, (shallow_logits, mid_logits, deep_logits)
'''

class Classifier(nn.Module):
    """Classifier for single feature input"""
    def __init__(self, in_features, out_features, hparams=None):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.hparams = hparams  # Store for compatibility, unused in forward

    def forward(self, x):
        return self.fc(x)

class WholeFish(nn.Module):
    def __init__(self, input_shape, num_classes, hparams, weights=None):
        super(WholeFish, self).__init__()
        featurizer = Featurizer(input_shape, hparams)
        classifier = Classifier(
            featurizer.n_outputs,
            num_classes,
            hparams['nonlinear_classifier'])
        self.net = nn.Sequential(
            featurizer, classifier
        )
        if weights is not None:
            self.load_state_dict(copy.deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(copy.deepcopy(weights))

    def forward(self, x):
        return self.net(x)