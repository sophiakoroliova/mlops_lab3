import torch
from torchvision import datasets, transforms
import pandas as pd
import logging
import os

# Configure logging to show informative messages during pipeline execution
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_data():
    """
        Stage 1: Download raw MNIST dataset and save it in a DVC-tracked format.

        This stage:
        - Downloads the MNIST dataset using torchvision
        - Saves raw data as .pt files (so DVC can track them)
        - Creates a dataset registry table (CSV) with metadata
        """
    logging.info("=== Data Download Stage Started ===")

    # Create directory for raw data
    os.makedirs("data/raw", exist_ok=True)

    # Define transformation (convert images to tensors)
    transform = transforms.ToTensor()

    # Download training and test datasets
    train_dataset = datasets.MNIST(root="data/raw", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="data/raw", train=False, download=True, transform=transform)

    # Save raw data in PyTorch format (DVC will track these files)
    torch.save({"data": train_dataset.data, "targets": train_dataset.targets}, "data/raw/train.pt")
    torch.save({"data": test_dataset.data, "targets": test_dataset.targets}, "data/raw/test.pt")

    # Create dataset registry table
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