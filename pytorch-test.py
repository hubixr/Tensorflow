import torch
import torch.nn as nn
import torch.optim as optim
import time

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Function to train the model
def train_model(device, model, criterion, optimizer, x_train, y_train, num_epochs):
    model.to(device)
    x_train = x_train.to(device)
    y_train = y_train.to(device)

    start_time = time.time()
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(x_train)
        loss = criterion(outputs, y_train)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Training on {device} took {elapsed_time:.2f} seconds")
    return elapsed_time

# Hyperparameters
input_size = 10
hidden_size = 20
output_size = 5
learning_rate = 0.0001
num_epochs = 10

# Initialize the model, loss function, and optimizer
model = SimpleNN(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Dummy data for training
x_train = torch.randn(100, input_size)
y_train = torch.randint(0, output_size, (100,))

# Train on GPU (if available)
device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
time_gpu = train_model(device_gpu, model, criterion, optimizer, x_train, y_train, num_epochs)

# Reinitialize the model and optimizer for CPU training
model = SimpleNN(input_size, hidden_size, output_size)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train on CPU
device_cpu = torch.device("cpu")
time_cpu = train_model(device_cpu, model, criterion, optimizer, x_train, y_train, num_epochs)

# Compare times
print(f"Time on GPU: {time_gpu:.2f} seconds")
print(f"Time on CPU: {time_cpu:.2f} seconds")
