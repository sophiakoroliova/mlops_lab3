# MLOps Lab 3: DVC Pipeline

A DVC pipeline for automating ML workflows (MNIST + PyTorch) has been implemented.

## Pipeline Stages
1. **download** — downloads MNIST dataset and creates dataset registry table  
2. **train** — trains a simple neural network (depends on data + `params.yaml`)  
3. **evaluate** — calculates accuracy and saves metrics (depends on trained model)

Run the full pipeline:
```bash
dvc repro