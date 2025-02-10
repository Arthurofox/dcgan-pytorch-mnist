import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.utils.data import DataLoader

def get_dataloader(batch_size, image_size):
    """Returns a DataLoader for FashionMNIST with appropriate transformations."""
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize images to [-1, 1]
    ])

    dataset = dsets.FashionMNIST(
        root='./data',
        train=True,
        transform=transform,
        download=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2  # Adjust based on your machine
    )
    
    return dataloader
