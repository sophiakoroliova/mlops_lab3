import torch
from torchvision import datasets, transforms
import pandas as pd
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_data():
    logging.info("=== Data Download Stage Started ===")
    os.makedirs("data/raw", exist_ok=True)

    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root="data/raw", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="data/raw", train=False, download=True, transform=transform)

    # Сохраняем сырые данные (чтобы DVC мог их трекать)
    torch.save({"data": train_dataset.data, "targets": train_dataset.targets}, "data/raw/train.pt")
    torch.save({"data": test_dataset.data, "targets": test_dataset.targets}, "data/raw/test.pt")

    # Dataset registry table
    registry = pd.DataFrame([{
        "version": "1.0",
        "description": "MNIST handwritten digits from torchvision",
        "num_train": 60000,
        "num_test": 10000,
        "image_size": "28x28",
        "num_classes": 10
    }])
    registry.to_csv("data/dataset_registry.csv", index=False)

    logging.info("Data downloaded and saved to data/raw/")
    logging.info("=== Data Download Stage Finished ===")

if __name__ == "__main__":
    download_data()