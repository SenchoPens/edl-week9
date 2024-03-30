import torch
import torch.nn as nn

from models import resnet101_truncated, resnet101_random
from train import train


def task1(device, *args):
    torch.manual_seed(0xeba1)
    student = resnet101_truncated()
    student = student.to(device)
    criterion = nn.CrossEntropyLoss()
    def model_step(images, labels):
        outputs = student(images)
        loss = criterion(outputs, labels)
        loss.backward()
        return loss.item()
    accuracy, epoch = train(device, student, model_step, *args)
    return student, accuracy, epoch


if __name__ == '__main__':
    from dataset import train_loader, test_loader
    model, accuracy, epoch, loss = task1('cuda', train_loader, test_loader)
    print(accuracy, epoch, loss)
    torch.save(model.state_dict(), 'task1.pt')
