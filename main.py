import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from models.transformer_model import TimeSeriesTransformer
from utils.preprocessing import prepare_data
from utils.plotting import plot_loss, plot_predictions
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# ---- Hyperparameters ----
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3
SEQ_LEN = 24
TARGET = "CO(GT)"
INPUT_COLS = ["CO(GT)"]  # Univariate for now

# ---- Prepare data ----
X_train, y_train, X_val, y_val, scaler = prepare_data(
    filepath="data/AirQualityUCI.csv",
    target_column=TARGET,
    input_columns=INPUT_COLS
)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)

# ---- Initialize model ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TimeSeriesTransformer(
    input_size=len(INPUT_COLS),
    d_model=64,
    nhead=4,
    num_layers=1,
    dropout=0.1
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ---- Training loop ----
train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss / len(train_loader))

    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            val_loss += loss.item()
        val_losses.append(val_loss / len(val_loader))

    print(f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

# ---- Evaluate and plot ----
model.eval()
with torch.no_grad():
    preds = []
    for xb, _ in val_loader:
        xb = xb.to(device)
        pred = model(xb)
        preds.append(pred.cpu())
    predictions = torch.cat(preds, dim=0)

# Plot loss and predictions
plot_loss(train_losses, val_losses)
plot_predictions(y_val.numpy(), predictions.numpy())

# Calculate metrics
mse = mean_squared_error(y_val.numpy(), predictions.numpy())
mae = mean_absolute_error(y_val.numpy(), predictions.numpy())
print(f"\nFinal Test MSE: {mse:.4f}")
print(f"Final Test MAE: {mae:.4f}")
