import torch
import torch.nn as nn
import torch.optim as optim

# 1. Create dummy data (y = 2x + 1)
X = torch.randn(100, 1)
y = 2 * X + 1

# 2. Define model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = SimpleNet()

# 3. Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 4. Training loop
for epoch in range(100):
    optimizer.zero_grad()
    predictions = model(X)
    loss = criterion(predictions, y)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

# 5. Test prediction
test_input = torch.tensor([[5.0]])
prediction = model(test_input)

print("\nTest Input: 5.0")
print("Predicted Output:", prediction.item())
