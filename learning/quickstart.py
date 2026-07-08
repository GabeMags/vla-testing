# I created this file to just get some basic quick start code going because I was having trouble keeping this research going independently. Best to start small and take small steps than grand ones. -GM

# I originally started with imageclassifiertutorial.py but burnt out. Hoping to get some traction with this basic tutorial from https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

# 6/20/26

import torch
from torch import nn
from torch.utils.data import DataLoader # DataLoader wraps an iterable around the Dataset
from torchvision import datasets # Dataset stores the samples and their corresponding labels
from torchvision.transforms import v2

# 7/3/26
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)

#Download test data from open datasets
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)

batch_size = 64

#Create data loaders
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

#---- Creating Models ------

#We're using the GPU to accelerate the forwarding of data through the NN- else if none then use the cpu
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            #nn.Linear(in_features, out_features) creates a fully connected layer — the most basic building block of a neural network.
            nn.Linear(28*28, 512), 
            nn.ReLU(), #Rectified Linear Unit
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
model = NeuralNetwork().to(device)
print(model)



#---- Optimizing Model Parameters ------

# To train a model, we need a loss function and an optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# each training loop, we calculate the loss and use it as a way to adjust each neuron for the next pass
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

# we also check the model's performance against the test dataset to ensure it is learning
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# we print the loss during each epoch and we want to see it go down over time
epochs = 5

for t in range(epochs):
    print(f"Epoch {t+1}\n------------------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)

print("Done!")

# save the model by serializing the internal state dictionary
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

