import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import gzip
import pickle
import random
data = pickle.load(gzip.open('mnist_light.pkl.gz'),encoding='latin1')

train_data = torch.Tensor(data[0][0])
train_data_label = torch.Tensor(data[0][1])
test_data = torch.Tensor(data[1][0])
test_data_label = torch.Tensor(data[1][1])

train_dataset = torch.utils.data.TensorDataset(train_data, train_data_label)
test_dataset = torch.utils.data.TensorDataset(test_data, test_data_label)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

def test(net):
    correct = 0
    total = 0
    with torch.no_grad():
        # Enumerate the whole testing set
        for d in test_loader:
            image, label = d
            # Get the net prediction
            output = net(image)
            total += 1
            # Add 1 if correct, 0 otherwise
            correct += int(np.argmax(output)) == (int(label[0]) if label[0].numpy().shape == () else int(np.argmax(label)))
    # Return the accuracy of the net on the test data
    return correct / total

def fit(net, optimizer, criterion):
    # For all epochs
    for e in range(epochs):
        print('epoch ' + str(e + 1) + ' starts')
        # Enumerate the whole training set
        for i, d in enumerate(train_loader, 0):
            inputs, labels = d
            # Reset the parameters gradients
            optimizer.zero_grad()
            # Get the net prediction
            outputs = net(inputs)
            # Compute the loss
            loss = criterion(outputs, labels)
            # Back propagate the gradient
            loss.backward()
            optimizer.step()
            # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            if not labels[0].numpy().shape == ():
                _, labels = torch.max(labels, 1)


class LinearNet(nn.Module):
    def __init__(self, f_s, a):
        super(LinearNet, self).__init__()
        # Define the activation function
        self.act = a
        # Define an array that will hold the layers's Torch variables
        self.layers = []
        # So that PyTorch can correctly detect all the variables, we first need 
        # to initialize the variables and then add them to the array for later use
        for i in range(len(f_s) - 1):
            # Initialise the Torch variable of the current layer
            exec('self.l_' + str(i) + ' = nn.Linear(f_s[' + str(i) + '], f_s[' + str(i + 1) + '])')
            # Add the new variable to the array
            exec('self.layers.append(self.l_' + str(i) + ')')
        
    def forward(self, x):
        # Goes through each layer in a feed forward manner
        # Gives the output through the activation function for 
        # every layer except for the last one
        for l in self.layers[:-1]:
            x = self.act(l(x))
        x = self.layers[-1](x)
        return x

# Parameters
learning_rate = .001
epochs = 1
# The array where we define the structure of the net
# We can add as many layers of any size as we want
features_sizes = [784, 128, 64, 10]
# Define the chosen activation function
activation_fn = torch.nn.ReLU()
# Define the chosen loss function
criterion = nn.MSELoss()

# Define the net and optimizer
linear_net = LinearNet(features_sizes, activation_fn)
optimizer = torch.optim.Adam(linear_net.parameters(), lr=learning_rate)

# Fit the net
fit(linear_net, optimizer, criterion)

# Print the accuracy on the test set
linear_net.eval()
acc = test(linear_net)
print('Accuracy: ' + str(acc))