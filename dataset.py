import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset


# Define CIFAR-10 data loaders
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # Resize to match ImageNet size
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, pin_memory=True, num_workers=16)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, pin_memory=True, num_workers=16)

num_images_to_keep = 2000
toy_train_dataset = Subset(train_dataset, range(min(num_images_to_keep, 50_000)))
toy_test_dataset = Subset(test_dataset, range(min(num_images_to_keep, 10_000)))

toy_train_loader = DataLoader(toy_train_dataset, batch_size=128, shuffle=True)
toy_test_loader = DataLoader(toy_test_dataset, batch_size=128, shuffle=False)
