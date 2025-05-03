import matplotlib.pyplot as plt

def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_predictions(y_true, y_pred, num_samples=100):
    plt.figure(figsize=(10, 5))
    plt.plot(y_true[:num_samples], label='Actual')
    plt.plot(y_pred[:num_samples], label='Predicted')
    plt.xlabel('Time Step')
    plt.ylabel('CO Concentration')
    plt.title('Predicted vs Actual Values')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
