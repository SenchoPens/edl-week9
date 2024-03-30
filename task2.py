import torch
import torch.nn as nn

from models import resnet101_truncated, kd_loss, resnet101_random
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
        return loss

    accuracy, epoch, loss = train(device, student, model_step, *args)
    return student, accuracy, epoch, loss


if __name__ == '__main__':
    from dataset import train_loader, test_loader
    teacher = resnet101_random()
    teacher.load_state_dict(torch.load('finetuned_resnet101.pt'))
    model, accuracy, epoch, loss = task2('cuda', teacher, train_loader, test_loader, 0.01)
    print(accuracy, epoch, loss)
    torch.save(model.state_dict(), 'task2.pt')
