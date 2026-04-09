import torch
import torch.nn as nn
import yaml
import json
import logging

# Configurate logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate():
    """
    Stage 3: Evaluate the trained model and save performance metrics.

    Dependencies:
    - trained model (model/model.pth)
    - params.yaml (for model architecture)
    - test data (data/raw/test.pt)

    Output:
    metrics.json (contains accuracy)
    """
    logging.info("=== Evaluation Stage Started ===")

    # Load hyperparameters
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    # Define the same model architecture
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(28*28, params["model"]["hidden_dim"])
            self.fc2 = nn.Linear(params["model"]["hidden_dim"], 10)
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Load trained model
    model = Net()
    model.load_state_dict(torch.load("models/model.pth"))
    model.eval()

    # Load test data
    test_pt = torch.load("data/raw/test.pt")
    X_test = test_pt["data"].float().view(-1, 28*28) / 255.0
    y_test = test_pt["targets"]

    # Evaluate
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test).float().mean().item()

    # Save metrics
    metrics = {"accuracy": round(accuracy * 100, 2)}
    with open("metrics.json", "w") as f:
        json.dump(metrics, f)

    logging.info(f"Accuracy: {metrics['accuracy']}%")
    logging.info("=== Evaluation Stage Finished ===")

if __name__ == "__main__":
    evaluate()