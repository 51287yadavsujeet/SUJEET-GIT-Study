import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)

print("PyTorch Demo: Training a Neural Network")
print("=" * 50)

# 1. Generate synthetic data (simple classification problem)
print("\n1. Generating synthetic data...")
n_samples = 1000
X = torch.randn(n_samples, 2)  # 2D input features
y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).long()  # Circular decision boundary

# Split into train and test
train_size = int(0.8 * n_samples)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# 2. Define a neural network
print("\n2. Building neural network...")


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = SimpleNet()
print(model)

# 3. Set up loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4. Training loop
print("\n3. Training the model...")
epochs = 50
losses = []

for epoch in range(epochs):
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    losses.append(avg_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

# 5. Evaluate the model
print("\n4. Evaluating the model...")
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    _, predicted = torch.max(test_outputs, 1)
    accuracy = (predicted == y_test).float().mean()
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

# 6. Visualize results
print("\n5. Creating visualizations...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot training loss
ax1.plot(losses)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss Over Time')
ax1.grid(True)

# Plot decision boundary
with torch.no_grad():
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = torch.meshgrid(torch.linspace(x_min, x_max, 100),
                            torch.linspace(y_min, y_max, 100), indexing='ij')
    grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    predictions = model(grid).argmax(dim=1).reshape(xx.shape)

ax2.contourf(xx.numpy(), yy.numpy(), predictions.numpy(), alpha=0.3, cmap='RdYlBu')
ax2.scatter(X_test[:, 0].numpy(), X_test[:, 1].numpy(),
            c=y_test.numpy(), cmap='RdYlBu', edgecolors='k')
ax2.set_xlabel('Feature 1')
ax2.set_ylabel('Feature 2')
ax2.set_title('Decision Boundary')

plt.tight_layout()
plt.savefig('pytorch_demo_results.png', dpi=150, bbox_inches='tight')
print("Visualization saved as 'pytorch_demo_results.png'")

print("\n" + "=" * 50)
print("Demo complete! Key PyTorch features demonstrated:")
print("✓ Tensor operations")
print("✓ Neural network definition (nn.Module)")
print("✓ DataLoader for batching")
print("✓ Training loop (forward, backward, optimize)")
print("✓ Model evaluation")
print("=" * 50)