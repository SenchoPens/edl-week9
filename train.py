import torch
import torch.optim as optim
from tqdm.auto import tqdm


def generator():
    i = 0
    while True:
        yield i
        i += 1


def train(device, model, model_step, train_loader, test_loader, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    prev_accuracy = 0.0
    stabilized_epochs = 0
    loss = None
    pbar = tqdm(range(generator()), desc="epoch")
    for epoch in pbar:
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = model_step(images, labels)
            optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = correct / total
            pbar.set_description(f'{loss=}, {accuracy=}')
            
            # Check if accuracy stabilizes
            if abs(accuracy - prev_accuracy) < 0.01:
                stabilized_epochs += 1
                if stabilized_epochs >= 2:
                    print(f"Accuracy stabilized to {accuracy:.4f} at epoch {epoch + 1}")
                    break
            else:
                stabilized_epochs = 0
            
            prev_accuracy = accuracy
    return prev_accuracy, epoch, loss
