import torch
from torchvision import datasets, transforms

# Define transformations (same as in train.py)
# This is needed to load the dataset object correctly
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # Standard MNIST normalization
])

# Load the training dataset (download if not already present)
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Load the test dataset (download if not already present)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Count the number of '0's in the training set
# train_dataset.targets is a tensor containing all the labels
# .eq(0) creates a boolean tensor (True where the label is 0, False otherwise)
# .sum() counts the number of True values
num_zeros_train = train_dataset.targets.eq(1).sum().item()

# Count the number of '0's in the test set
num_zeros_test = test_dataset.targets.eq(1).sum().item()

# Print the counts
print(f"Number of '1's in the training set: {num_zeros_train}")
print(f"Number of '1's in the test set: {num_zeros_test}")
print(f"Total number of '1's in MNIST: {num_zeros_train + num_zeros_test}")

