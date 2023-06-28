import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.fc = nn.Linear(10, 1)  # Example linear layer

    def forward(self, x):
        return self.fc(x)

checkpoint_path = "checkpoint.pth"
checkpoint = torch.load(checkpoint_path)

# Create a dataset and data loader (dummy data for demonstration)
# Replace with your own dataset and data loader
train_dataset = torch.randn(100, 10)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)

model = MyNetwork()
model.load_state_dict(checkpoint['model_state_dict'])

optimizer = optim.SGD(model.parameters(), lr=0.1)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
warmup_factor = 1.0 / 1000
warmup_iters = min(1000, len(train_loader) - 1)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.5, end_factor=1, total_iters=warmup_iters)
# scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.0556, total_iters=warmup_iters, last_epoch=checkpoint['scheduler_state_dict']['last_epoch']-1)



print(scheduler.get_last_lr()[0])




# Training loop
for epoch in range(1):
    for batch_idx, data in enumerate(train_loader):
        if batch_idx <= 4:
            continue
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
        scheduler.step()

        print(f"Epoch [{epoch}/{10}], Batch [{batch_idx}/{len(train_loader)}], "
              f"Loss: {loss.item()}, Learning Rate: {scheduler.get_last_lr()[0]}")
        
    #     torch.save({
    #     'epoch': epoch,
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'scheduler_state_dict': scheduler.state_dict()
    # }, checkpoint_path)
