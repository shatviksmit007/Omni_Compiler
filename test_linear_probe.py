import torch
import torch.nn as nn
import torch.optim as optim

# 1. Define the Linear Probe for the Shoes dataset
# Assuming CLIP outputs 512D embeddings. Update num_classes for your exact shoe types.
num_shoe_classes = 3 
linear_probe = nn.Linear(512, num_shoe_classes)

# 2. Setup Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(linear_probe.parameters(), lr=1e-3)

# 3. Mock Dataset (Replace with actual CLIP embeddings from your shoes dataset)
# Simulating 100 shoe images already converted to 512D embeddings
mock_embeddings = torch.randn(100, 512) 
mock_labels = torch.randint(0, num_shoe_classes, (100,))

# 4. Training Loop (Extremely fast CPU-friendly training)
epochs = 51
for epoch in range(epochs):
    optimizer.zero_grad()
    
    # Forward pass: just the linear layer
    outputs = linear_probe(mock_embeddings) 
    loss = criterion(outputs, mock_labels)
    
    # Backpropagate only through the linear head
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

# 5. Save the weights for the CLI demo
torch.save(linear_probe.state_dict(), "clip_probe_shoes.pth")
print("Saved shoes weights to clip_probe_shoes.pth")