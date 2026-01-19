import torch
import torch.nn as nn

# 1. Create Data: y = 2x
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

# 2. Define a Simple Model
# nn.Linear(input_size, output_size)
model = nn.Linear(1, 1)

# 3. Define Loss and Optimizer
loss_fn = nn.MSELoss()  # Mean Squared Error
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # Stochastic Gradient Descent

# 4. Training Loop
print("Training...")
for epoch in range(100):
    # Forward pass: Predict y by passing X to the model
    y_pred = model(X)

    # Compute loss
    loss = loss_fn(y_pred, y)

    # Backward pass: Compute gradients and update weights
    optimizer.zero_grad() # Clear previous gradients
    loss.backward()      # Backpropagation
    optimizer.step()     # Update weights

    if (epoch+1) % 20 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

# 5. Test the Model
test_val = torch.tensor([[5.0]])
predicted = model(test_val)
print(f'\nPrediction for x=5: {predicted.item():.4f} (Expected: 10.0)')