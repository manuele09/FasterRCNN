import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

# Define the neural network
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.fc = nn.Linear(10, 1)  # Example linear layer

    def forward(self, x):
        return self.fc(x)

checkpoint_path = "checkpoint.pth"

# Create a dataset and data loader (dummy data for demonstration)
# Replace with your own dataset and data loader
train_dataset = torch.randn(100, 10)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)



# Initialize the network, optimizer, and scheduler
model = MyNetwork()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training loop
for epoch in range(1):
    for batch_idx, data in enumerate(train_loader):
        inputs = data  # Example input data
        targets = torch.randn(inputs.size(0))  # Example target values

        # Forward pass
        outputs = model(inputs)
        loss = nn.MSELoss()(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Step the scheduler

        print(f"Epoch [{epoch}/{10}], Batch [{batch_idx}/{len(train_loader)}], "
              f"Loss: {loss.item()}")
        

print("Training complete.")



