import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_dataloaders(batch_size=128):
    # The exact normalization values requested by the lab
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    # Load the official 60k training dataset
    full_train_dataset = datasets.FashionMNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )

    # Split 60k into 55k train and 5k validation (seeded for reproducibility)
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        full_train_dataset, [55000, 5000], generator=generator
    )

    # Load the official 10k test dataset
    test_dataset = datasets.FashionMNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )

    # Create the DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()
    
    print(f"Train batches: {len(train_loader)} (Size: {len(train_loader.dataset)})")
    print(f"Val batches:   {len(val_loader)} (Size: {len(val_loader.dataset)})")
    print(f"Test batches:  {len(test_loader)} (Size: {len(test_loader.dataset)})")
    
    # Grab one batch to verify the shapes
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}") # Should be [128, 1, 28, 28]