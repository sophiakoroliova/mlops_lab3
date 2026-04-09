import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import yaml
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train():
    logging.info("=== Training Stage Started ===")
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    # Загружаем данные
    train_pt = torch.load("data/raw/train.pt")
    X_train = train_pt["data"].float().view(-1, 28*28) / 255.0
    y_train = train_pt["targets"]
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=params["batch_size"], shuffle=True)

    # Простая нейросеть
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(28*28, params["model"]["hidden_dim"])
            self.fc2 = nn.Linear(params["model"]["hidden_dim"], 10)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=params["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(params["epochs"]):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        logging.info(f"Epoch {epoch+1}/{params['epochs']} completed")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/model.pth")
    logging.info("Model saved to models/model.pth")
    logging.info("=== Training Stage Finished ===")

if __name__ == "__main__":
    train()