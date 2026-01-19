import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# -------------------------------
# 1. Create Dataset
# -------------------------------
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# -------------------------------
# 2. Define Neural Network
# -------------------------------
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleNN()

# -------------------------------
# 3. Loss & Optimizer
# -------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# -------------------------------
# 4. Training Loop
# -------------------------------
epochs = 200

for epoch in range(epochs):
    model.train()

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# -------------------------------
# 5. Evaluation
# -------------------------------
model.eval()
with torch.no_grad():
    predictions = torch.argmax(model(X_test), dim=1)
    accuracy = (predictions == y_test).float().mean()

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# -------------------------------
# 6. Visualization
# -------------------------------
plt.scatter(X_test[:,0], X_test[:,1], c=predictions, cmap="coolwarm")
plt.title("PyTorch Classification Demo")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
