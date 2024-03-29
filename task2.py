import torch
import torch.nn as nn

from models import resnet101_truncated, kd_loss
from train import train


def task2(device, teacher, *args):
    torch.manual_seed(0xEBA1)
    student = resnet101_truncated()
    student = student.to(device)
    teacher = teacher.to(device)
    criterion = nn.CrossEntropyLoss()

    def model_step(images, labels):
        student_logits = student(images)
        with torch.no_grad():
            teacher_logits = teacher(images)
        label_loss = criterion(student_logits, labels)
        loss = kd_loss(teacher_logits, student_logits, label_loss)
        loss.backward()
        return loss.item()

    accuracy, epoch = train(device, student, model_step, *args)
    return student, accuracy, epoch
