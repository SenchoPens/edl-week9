import torch
import torch.nn as nn
from torch.nn.functional import mse_loss

from models import resnet101_truncated, kd_loss, resnet101_random
from train import train


def get_activation(name, acts):
    def hook(model, input, output):
        acts[name] = output
    return hook


def capture_layers(model, acts):
    model.layer1.register_forward_hook(get_activation('layer1', acts))
    model.layer2.register_forward_hook(get_activation('layer2', acts))
    model.layer4.register_forward_hook(get_activation('layer4', acts))


def feature_loss(student_acts, teacher_acts):
    loss = 0
    for layer in ('layer1', 'layer2', 'layer4'):
        student_act = student_acts[layer]
        teacher_act = teacher_acts[layer]
        loss += mse_loss(student_act, teacher_act)
    return loss / 3


def task3(device, teacher, *args):
    torch.manual_seed(0xEBA1)
    student = resnet101_truncated()
    student = student.to(device)
    teacher = teacher.to(device)
    student_activations = {}
    capture_layers(student, student_activations)
    teacher_activations = {}
    capture_layers(teacher, teacher_activations)
    criterion = nn.CrossEntropyLoss()

    def model_step(images, labels):
        student_logits = student(images)
        with torch.no_grad():
            teacher_logits = teacher(images)
        label_loss = criterion(student_logits, labels)
        kd_l = kd_loss(teacher_logits, student_logits, label_loss)
        feature_l = feature_loss(student_activations, teacher_activations)
        # print(kd_l, feature_l)
        loss = 0.66 * kd_l + 0.34 * feature_l
        loss.backward()
        return loss

    accuracy, epoch, loss = train(device, student, model_step, *args)
    return student, accuracy, epoch, loss


if __name__ == '__main__':
    from dataset import train_loader, test_loader
    teacher = resnet101_random()
    teacher.load_state_dict(torch.load('finetuned_resnet101.pt'))
    model, accuracy, epoch, loss = task3('cuda', teacher, train_loader, test_loader)
    print(accuracy, epoch, loss)
    torch.save(model.state_dict(), 'task3.pt')
