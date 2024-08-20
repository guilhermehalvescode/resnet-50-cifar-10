import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard

# Data Preparation
transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ]
)

trainset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform_train
)
testset = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform_test
)

validation_split = 0.1
shuffle_dataset = True
random_seed = 42

dataset_size = len(trainset)
indices = list(range(dataset_size))
split = int(validation_split * dataset_size)

if shuffle_dataset:
    torch.manual_seed(random_seed)
    torch.random.manual_seed(random_seed)
    torch.random.manual_seed(random_seed)
    torch.utils.data.random_split(trainset, [split, dataset_size - split])

train_sampler = SubsetRandomSampler(indices[split:])
valid_sampler = SubsetRandomSampler(indices[:split])

trainloader = data.DataLoader(
    trainset, batch_size=128, sampler=train_sampler, num_workers=2
)
validloader = data.DataLoader(
    trainset, batch_size=128, sampler=valid_sampler, num_workers=2
)
testloader = data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Model Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

# Early Stopping Parameters
patience = 5
best_valid_loss = float("inf")
epochs_no_improve = 0

# TensorBoard SummaryWriter
writer = SummaryWriter("runs/resnet50_cifar10")  # Create a SummaryWriter

# Initialize batch counters
train_batch_num = 0
valid_batch_num = 0

# Training Loop with Early Stopping and Detailed TensorBoard Logging
num_epochs = 25

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Log per-batch metrics for training
        train_loss = loss.item()
        train_acc = 100.0 * predicted.eq(labels).sum().item() / labels.size(0)
        writer.add_scalar("Batch_Loss/Train", train_loss, train_batch_num)
        writer.add_scalar("Batch_Accuracy/Train", train_acc, train_batch_num)
        train_batch_num += 1

    train_loss = running_loss / len(trainloader.sampler)
    train_acc = 100.0 * correct / total

    # Validation Loop
    model.eval()
    valid_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in validloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            valid_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Log per-batch metrics for validation
            batch_valid_loss = loss.item()
            batch_valid_acc = 100.0 * predicted.eq(labels).sum().item() / labels.size(0)
            writer.add_scalar(
                "Batch_Loss/Validation", batch_valid_loss, valid_batch_num
            )
            writer.add_scalar(
                "Batch_Accuracy/Validation", batch_valid_acc, valid_batch_num
            )
            valid_batch_num += 1

    valid_loss = valid_loss / len(validloader.sampler)
    valid_acc = 100.0 * correct / total

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], "
        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
        f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.2f}%"
    )

    # Log metrics to TensorBoard for each epoch
    writer.add_scalar("Epoch_Loss/Train", train_loss, epoch + 1)
    writer.add_scalar("Epoch_Loss/Validation", valid_loss, epoch + 1)
    writer.add_scalar("Epoch_Accuracy/Train", train_acc, epoch + 1)
    writer.add_scalar("Epoch_Accuracy/Validation", valid_acc, epoch + 1)

    # Early Stopping Check
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), "best_model.pth")  # Save the best model
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

# Testing Loop
model.load_state_dict(torch.load("best_model.pth"))  # Load the best model
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

test_acc = 100.0 * correct / total
print(f"Test Accuracy: {test_acc:.2f}%")

# Close the TensorBoard writer
writer.close()
