import torch
import torch.nn as nn

from models import resnet101_pretrained
from train import train


def finetune_resnet101(device, *args):
    torch.manual_seed(0xeba1)
    model = resnet101_pretrained()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    def model_step(images, labels):
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        return loss.item()
    accuracy, epoch, loss = train(device, model, model_step, *args)
    return model, accuracy, epoch, loss


# num_epochs = 10
# teacher_model = finetune_resnet101(num_epochs, 'cuda:0')

if __name__ == '__main__':
    from dataset import train_loader, test_loader
    model, accuracy, epoch, loss = finetune_resnet101('cuda', train_loader, test_loader)
    print(accuracy, epoch, loss)
    torch.save(model.state_dict(), 'finetuned_resnet101.pt')
