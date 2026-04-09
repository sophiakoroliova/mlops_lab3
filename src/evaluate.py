import torch
import torch.nn as nn
import yaml
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate():
    logging.info("=== Evaluation Stage Started ===")
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    # Загружаем модель
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
    model.load_state_dict(torch.load("models/model.pth"))
    model.eval()

    # Загружаем test data
    test_pt = torch.load("data/raw/test.pt")
    X_test = test_pt["data"].float().view(-1, 28*28) / 255.0
    y_test = test_pt["targets"]

    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test).float().mean().item()

    metrics = {"accuracy": round(accuracy * 100, 2)}
    with open("metrics.json", "w") as f:
        json.dump(metrics, f)

    logging.info(f"Accuracy: {metrics['accuracy']}%")
    logging.info("=== Evaluation Stage Finished ===")

if __name__ == "__main__":
    evaluate()