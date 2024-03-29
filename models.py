import torch
import torch.nn as nn
import torchvision.models as models


class Identity(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x


def kd_loss(teacher_logits, student_logits, label_loss, T=2.0):
    soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
    soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)
    soft_targets_loss = (
        torch.sum(-soft_targets * soft_prob) / soft_prob.size()[0] * (T**2)
    )
    return 0.5 * soft_targets_loss + 0.5 * label_loss


def adjust_resnet101_for_cifar10(model):
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = Identity()
    return model


def resnet101_pretrained():
    # model = models.resnet101(models.ResNet101_Weights.IMAGENET1K_V2)
    model = models.resnet101(pretrained=True)
    model = adjust_resnet101_for_cifar10(model)
    return model


def resnet101_truncated():
    model = models.resnet101()  # _resnet(Bottleneck, [3, 4, 23, 3])
    model = adjust_resnet101_for_cifar10(model)
    model.layer3 = nn.Conv2d(512, 1024, kernel_size=2, stride=2, bias=False)
    return model
