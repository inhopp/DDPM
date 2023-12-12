import torch
from torchvision import datasets
import torchvision.transforms as transforms

def generate_loader(phase, opt):
    img_size = opt.input_size

    if phase == 'train':
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1),
        ])
    
    dataset = datasets.FashionMNIST(root="./datasets", train=True, download=True, transform=transform)

    kwargs = {
        "batch_size": opt.batch_size,
        "num_workers": opt.num_workers,
        "shuffle": True,
        "drop_last": True,
    }

    return torch.utils.data.DataLoader(dataset, **kwargs)