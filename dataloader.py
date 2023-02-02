import os
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms


# Load data
def get_data_loader(args):
    data_dir = args.data
    batch_size = args.batch_size
    transform_train = transforms.Compose([
        torchvision.transforms.Resize(size=(512, 512)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomCrop(size=(448, 448)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    # transform_strong = transforms.Compose([
    #     torchvision.transforms.Resize(size=(512, 512)),
    #     torchvision.transforms.RandomHorizontalFlip(),
    #     torchvision.transforms.RandomCrop(size=(448, 448)),
    #     transforms.RandomGrayscale(p=0.2),
    #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    transform_test = transforms.Compose([
        torchvision.transforms.Resize(size=(512, 512)),
        torchvision.transforms.CenterCrop(size=(448, 448)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    train_data = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform_train)
    # train_strong_data = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform_strong)
    test_data = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform_test)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    # train_strong_loader = DataLoader(train_strong_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    return train_loader, test_loader
