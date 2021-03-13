import torch.hub
from torch.hub import load_state_dict_from_url
import torch.nn as nn
import torch
from torchvision.models.resnet import ResNet, Bottleneck

def _resnext101_32xxxd_wsl(name, num_classes=1000):
    model = torch.hub.load('facebookresearch/WSL-Images', name)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def resnext101_32x8d_wsl(num_classes=1000):
    return _resnext101_32xxxd_wsl('resnext101_32x8d_wsl', num_classes=num_classes)


def resnext101_32x16d_wsl(num_classes=1000):
    return _resnext101_32xxxd_wsl('resnext101_32x16d_wsl', num_classes=num_classes)


def resnext101_32x32d_wsl(num_classes=1000):
    return _resnext101_32xxxd_wsl('resnext101_32x32d_wsl', num_classes=num_classes)


def resnext101_32x48d_wsl(num_classes=1000):
    return _resnext101_32xxxd_wsl('resnext101_32x48d_wsl', num_classes=num_classes)






#def _resnext(arch, block, layers, pretrained, progress, **kwargs):
#    model = ResNet(block, layers, **kwargs)
#    state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
#    model.load_state_dict(state_dict)
#    return model


#def resnext101_32x8d_wsl(progress=True, **kwargs):
    """Constructs a ResNeXt-101 32x8 model pre-trained on weakly-supervised data
    and finetuned on ImageNet from Figure 5 in
    `"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_
    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
#    kwargs['groups'] = 32
#    kwargs['width_per_group'] = 8
#    return _resnext('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], True, progress, **kwargs)


#def resnext101_32x16d_wsl(progress=True, **kwargs):
    """Constructs a ResNeXt-101 32x16 model pre-trained on weakly-supervised data
    and finetuned on ImageNet from Figure 5 in
    `"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_
    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
#    kwargs['groups'] = 32
#    kwargs['width_per_group'] = 16
#    return _resnext('resnext101_32x16d', Bottleneck, [3, 4, 23, 3], True, progress, **kwargs)


#def resnext101_32x32d_wsl(progress=True, **kwargs):
    """Constructs a ResNeXt-101 32x32 model pre-trained on weakly-supervised data
    and finetuned on ImageNet from Figure 5 in
    `"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_
    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
#    kwargs['groups'] = 32
#    kwargs['width_per_group'] = 32
#    return _resnext('resnext101_32x32d', Bottleneck, [3, 4, 23, 3], True, progress, **kwargs)


#def resnext101_32x48d_wsl(progress=True, **kwargs):
    """Constructs a ResNeXt-101 32x48 model pre-trained on weakly-supervised data
    and finetuned on ImageNet from Figure 5 in
    `"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_
    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
#    kwargs['groups'] = 32
#    kwargs['width_per_group'] = 48
#    return _resnext('resnext101_32x48d', Bottleneck, [3, 4, 23, 3], True, progress, **kwargs)
