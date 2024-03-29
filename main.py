import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# Define CIFAR-10 data loaders
transform = transforms.Compose([
    # transforms.Resize((224, 224)),  # Resize to match ImageNet size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Define ResNet101 architecture
class ResNet101(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet101, self).__init__()
        # Define ResNet layers
        ...

    def forward(self, x):
        ...

# Task 1: Finetune ResNet101 on CIFAR10
def finetune_resnet101():
    model = ResNet101()
    # Replace classification layer
    model.fc = nn.Linear(model.fc.in_features, 10)
    # Initialize weights
    ...
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = correct / total
        print("Task 1 Test Accuracy: {:.2f}%".format(accuracy * 100))
    return model

# Task 2: Train the truncated ResNet101 on input data only
def train_student_only():
    model = ResNet101()
    # Remove layer3
    ...
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = correct / total
        print("Task 2 Test Accuracy: {:.2f}%".format(accuracy * 100))
    return model

# Task 3: Train the truncated ResNet101 with soft cross-entropy and MSE loss
def train_student_soft_distillation():
    teacher_model = finetune_resnet101()
    student_model = ResNet101()
    # Remove layer3
    ...
    # Define optimizer
    optimizer = optim.Adam(student_model.parameters(), lr=0.001)
    # Define losses
    criterion_ce = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()

    # Training loop
    for epoch in range(num_epochs):
        student_model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            student_outputs = student_model(images)
            teacher_outputs = teacher_model(images)
            ce_loss = criterion_ce(student_outputs, labels)
            mse_loss = 0
            for student_feat, teacher_feat in zip(student_features, teacher_features):
                mse_loss += criterion_mse(student_feat, teacher_feat)
            loss = ce_loss + mse_loss
            loss.backward()
            optimizer.step()

    # Evaluation
    student_model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = student_model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = correct / total
        print("Task 3 Test Accuracy: {:.2f}%".format(accuracy * 100))
    return student_model

# Training parameters
num_epochs = 10

# Task 1
teacher_model = finetune_resnet101()

# Task 2
student_model_only = train_student_only()

# Task 3
student_model_soft_distillation = train_student_soft_distillation()
